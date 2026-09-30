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
with np.load(repository / "tests/references/deterministic_rate_rows.npz") as historical:
    model = Subhalos(2.0)
    expected = np.stack([model.Na_calc(row, np.array([z]), 1e10, N_herm=1,
                                      sigmafac=0.0)[0]
                         for row, z in zip(historical["mvir"], historical["z"], strict=True)])
    destination = args.output / "deterministic_rate_rows.npz"
    np.savez(destination, mvir=historical["mvir"], z=historical["z"], expected=expected)
    report["runner_files"]["deterministic_rate_rows"] = hashlib.sha256(destination.read_bytes()).hexdigest()
(args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
