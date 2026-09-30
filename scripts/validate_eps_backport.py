"""Compare pinned migration/main sources with the EPS backport in fresh processes.

Run with the candidate environment (including matplotlib for standalone main):
  python scripts/validate_eps_backport.py --baseline-source OLD --main-source MAIN \
      --output docs/validation/eps-backport.json

This reads the reference checkouts; it never rewrites frozen catalog fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import numpy as np

BASELINE = "dcef1910d42cab940567be448ffc79b42d436802"
MAIN = "685bcfde7bf3ac5805492b56070151ba0415a9ca"
CORE = "23d01e8758a88b061b87de9e488c38ec89fd8e4f"


def cases():
    base = dict(M0=1e10, redshift=0.0, dz=0.5, zmax=1.0, N_ma=4,
                sigmalogc=0.128, N_herm=2, logmamin=6.0, logmamax=8.0,
                N_hermNa=3, sigmafac=0.0, profile_change=True)
    result = {"default": (2.0, base), "particle_mass": (4.0, base)}
    for name, changes in {
        "nonzero": dict(redshift=0.5, zmax=1.5),
        "profile_off": dict(profile_change=False),
        "deterministic": dict(N_hermNa=1, N_ma=3, zmax=1.5),
        "sigma_offset": dict(N_hermNa=1, N_ma=3, zmax=1.5, sigmafac=0.5),
    }.items():
        result[name] = (2.0, base | changes)
    population = base | dict(M0=1e12, zmax=3.0, N_ma=6, N_herm=3,
                             logmamin=8.0, logmamax=11.0, N_hermNa=200)
    for mass in (0.5, 2.0, 5.0):
        result[f"quadrature_{mass}"] = (mass, population)
    for sigma in (-0.5, 0.0, 0.5):
        result[f"deterministic_{sigma}"] = (2.0, population | dict(N_hermNa=1, sigmafac=sigma))
    result["zero_population"] = (0.5, base | dict(M0=1e12))
    return result


def snapshot(source, output):
    source = source.resolve()
    sys.path.insert(0, str(source / "src" if (source / "src").exists() else source))
    import sashimi_w
    arrays, records = {}, {}
    for name, (mass, options) in cases().items():
        record = {"mass_wdm": mass, "parameters": options}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                catalog = sashimi_w.subhalos(mass).rs_rhos_calc(**options)
            record["warnings"] = sorted(set(str(w.message) for w in caught))
            if not all(np.all(np.isfinite(x)) for x in catalog):
                raise ValueError("Non-finite catalog output")
            for i, column in enumerate(catalog):
                arrays[f"{name}_{i}"] = column
            weight = catalog[8] * catalog[9]
            record["survivor_count"] = float(np.sum(weight))
            record["bound_mass_fraction"] = float(np.sum(weight * catalog[4]) / options["M0"])
            record["status"] = "ok"
        except (ValueError, IndexError, FloatingPointError) as error:
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
        records[name] = record
    rate_cases = []
    for mass in (0.5, 2.0, 5.0):
        model = sashimi_w.subhalos(mass)
        for order in (1, 4, 64, 200):
            name = f"rate_{mass}_{order}"
            z = np.array([0.25, 0.5, 1.0, 2.0])
            masses = np.logspace(1, 11, 9)
            try:
                with np.errstate(invalid="raise", divide="raise", over="raise"):
                    arrays[name] = model.Na_calc(masses, z, 1e12, N_herm=order)
                rate_cases.append({"name": name, "status": "ok"})
            except (ValueError, IndexError, FloatingPointError) as error:
                rate_cases.append({"name": name, "status": "error", "error": str(error)})
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    output.write_text(json.dumps({"catalogs": records, "rates": rate_cases}, indent=2) + "\n")


def compare(reference, candidate):
    old, new = reference["catalogs"], candidate["catalogs"]
    records = {}
    for name in old:
        record = {"baseline_status": old[name]["status"], "candidate_status": new[name]["status"]}
        if old[name]["status"] == new[name]["status"] == "ok":
            for metric in ("survivor_count", "bound_mass_fraction"):
                a, b = old[name][metric], new[name][metric]
                record[metric] = {"baseline": a, "candidate": b,
                                  "relative_change": b / a - 1 if a else None}
        else:
            record["baseline_error"] = old[name].get("error")
            record["candidate_error"] = new[name].get("error")
        records[name] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--main-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.snapshot:
        snapshot(args.snapshot, args.output)
        return
    from itamae.provenance import source_revision
    installed_core = source_revision("itamae")
    if installed_core != CORE:
        raise ValueError(f"Expected installed ITAMAE {CORE}, received {installed_core}")
    root = Path(__file__).resolve().parents[1]
    for source, revision in ((args.baseline_source, BASELINE), (args.main_source, MAIN)):
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError(f"Expected {revision}, received {actual}")
    sources = {"baseline": args.baseline_source, "main": args.main_source, "candidate": root}
    snapshots, arrays = {}, {}
    with tempfile.TemporaryDirectory() as temporary:
        for label, source in sources.items():
            output = Path(temporary) / f"{label}.json"
            env = os.environ | {"MPLCONFIGDIR": str(Path(temporary) / "mpl"),
                                "XDG_CACHE_HOME": str(Path(temporary) / "cache")}
            subprocess.run([sys.executable, __file__, "--snapshot", str(source), "--output", str(output)],
                           check=True, env=env)
            snapshots[label] = json.loads(output.read_text())
            with np.load(output.with_suffix(".npz")) as data:
                arrays[label] = {k: data[k] for k in data.files}
        differences = {}
        for reference in ("baseline", "main"):
            differences[reference] = compare(snapshots[reference], snapshots["candidate"])
            for name in differences[reference]:
                if f"{name}_0" not in arrays[reference]:
                    continue
                metrics = []
                for i in range(10):
                    old, new = arrays[reference][f"{name}_{i}"], arrays["candidate"][f"{name}_{i}"]
                    metrics.append(float(np.max(np.abs(new.astype(float) - old.astype(float))) /
                                         max(np.max(np.abs(old)), np.finfo(float).tiny)))
                differences[reference][name]["tuple_peak_scaled_abs_errors"] = metrics
        rate_differences = {}
        for name in arrays["main"]:
            if name.startswith("rate_"):
                old, new = arrays["main"][name], arrays["candidate"][name]
                rate_differences[name] = float(np.max(abs(new - old)) / np.max(old))
    files = ["_physics.py", "_itamae_migration.py", "_itamae_variance.py"]
    report = {
        "baseline_revision": BASELINE, "standalone_main_revision": MAIN, "itamae_revision": CORE,
        "candidate_source_sha256": {f: hashlib.sha256((root / "src/sashimi_w" / f).read_bytes()).hexdigest()
                                    for f in files},
        "cases": {name: {"mass_wdm": mass, "parameters": p} for name, (mass, p) in cases().items()},
        "comparisons": differences, "main_rate_peak_scaled_abs_errors": rate_differences,
        "strict_rate_baseline": snapshots["baseline"]["rates"],
        "strict_rate_candidate": snapshots["candidate"]["rates"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
