"""Measure frozen-reference transport using the unchanged installed migration."""

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

BASELINE = "dcef1910d42cab940567be448ffc79b42d436802"
assert source_revision("sashimi-w") == BASELINE
repository = Path(__file__).resolve().parents[1]
module_path = Path(sashimi_w.__file__).resolve()
assert module_path.is_relative_to(repository / ".baseline-install")
root = repository / "tests/references/native-api-baseline"
report = {
    "source": BASELINE,
    "module_path": str(module_path),
    "python": sys.version,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "platform": platform.platform(),
    "cases": {},
}
for path in sorted(root.glob("*.json")):
    record = json.loads(path.read_text())
    assert hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() == record["sha256"]
    catalog = Subhalos(**record["constructor"]).rs_rhos_catalog_calc(**record["parameters"])
    fields = {}
    with np.load(path.with_suffix(".npz")) as saved:
        for key, actual in {**catalog.columns, **catalog.weights}.items():
            expected = saved["default__" + key]
            if actual.dtype.kind == "b":
                fields[key] = {"equal": bool(np.array_equal(actual, expected))}
            else:
                nonzero = expected != 0
                relative = np.abs((actual[nonzero] - expected[nonzero]) / expected[nonzero])
                fields[key] = {
                    "max_relative": float(relative.max()) if relative.size else 0.0,
                    "max_absolute": float(np.max(np.abs(actual - expected))),
                    "finite": bool(np.all(np.isfinite(actual))),
                    "zero_support_equal": bool(np.array_equal(actual == 0, expected == 0)),
                }
    report["cases"][path.stem] = fields
print(json.dumps(report, indent=2, sort_keys=True))
