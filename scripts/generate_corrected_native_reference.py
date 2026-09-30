"""Generate independent post-EPS migration catalogs from the installed pinned source."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from itamae.provenance import source_revision

import sashimi_w
from sashimi_w import Subhalos

CORRECTED = "65f25390f7d745d671818ba904c59b133ce1aa29"
CORE = "23d01e8758a88b061b87de9e488c38ec89fd8e4f"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
repository = Path(__file__).resolve().parents[1]
module_path = Path(sashimi_w.__file__).resolve()
assert source_revision("sashimi-w") == CORRECTED
assert source_revision("itamae") == CORE
assert module_path.is_relative_to(repository / ".corrected-install")
root = repository / "tests/references/native-api-baseline"
report = {
    "source": CORRECTED,
    "itamae_source": CORE,
    "role": "independently installed post-EPS migration; no native API",
    "module_path": str(module_path),
    "python": sys.version,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "platform": platform.platform(),
    "runner_files": {},
}
for path in sorted(root.glob("*.json")):
    record = json.loads(path.read_text())
    assert hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() == record["sha256"]
    catalog = Subhalos(**record["constructor"]).rs_rhos_catalog_calc(**record["parameters"])
    assert catalog.metadata["deterministic_scatter_anchor"] == "exact-z=1"
    destination = args.output / (path.stem + ".npz")
    np.savez(destination, **catalog.columns, **catalog.weights)
    report["runner_files"][path.stem] = hashlib.sha256(destination.read_bytes()).hexdigest()
# Preserve the original nonsquare inputs; regenerate only the independently
# evaluated one-dimensional row rates using this pinned post-EPS source.
historical_path = repository / "tests/references/deterministic_rate_rows.npz"
historical_meta = json.loads(historical_path.with_suffix(".json").read_text())
assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == historical_meta["sha256"]
with np.load(historical_path) as historical:
    model = Subhalos(2.0)
    expected = np.stack([model.Na_calc(row, np.array([z]), 1e10, N_herm=1,
                                      sigmafac=0.0)[0]
                         for row, z in zip(historical["mvir"], historical["z"], strict=True)])
    destination = args.output / "deterministic_rate_rows.npz"
    np.savez(destination, mvir=historical["mvir"], z=historical["z"], expected=expected)
    report["runner_files"]["deterministic_rate_rows"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    saved_path = repository / "tests/references/native-api-eps-baseline/deterministic_rate_rows.npz"
    saved_meta = json.loads(saved_path.with_suffix(".json").read_text())
    assert saved_meta["source"] == CORRECTED
    assert hashlib.sha256(saved_path.read_bytes()).hexdigest() == saved_meta["sha256"]
    with np.load(saved_path) as saved:
        np.testing.assert_array_equal(historical["mvir"], saved["mvir"])
        np.testing.assert_array_equal(historical["z"], saved["z"])
        np.testing.assert_array_equal(expected == 0, saved["expected"] == 0)
        nonzero = saved["expected"] != 0
        relative = np.abs(expected[nonzero] / saved["expected"][nonzero] - 1)
        report["nonsquare_row_transport"] = {
            "source": CORRECTED,
            "saved_sha256": saved_meta["sha256"],
            "max_relative": float(relative.max()) if relative.size else 0.0,
            "max_absolute": float(np.max(np.abs(expected - saved["expected"]))),
            "inputs_equal": True,
            "zero_support_equal": True,
            "runner_expected": expected.tolist(),
            "saved_expected": saved["expected"].tolist(),
        }
(args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
