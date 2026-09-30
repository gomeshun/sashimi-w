"""The one-history branch must distinguish redshift and mass axes."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sashimi_w import Subhalos


def test_nonsquare_grid_matches_independent_corrected_single_rows():
    path = Path(__file__).parent / "references/deterministic_rate_rows.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["source"] == "dcef1910d42cab940567be448ffc79b42d436802"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    # Historical arrays remain immutable, but the stable EPS arithmetic has a
    # separate corrected oracle rather than a relaxed old-reference tolerance.
    path = Path(__file__).parent / "references/native-api-eps-baseline/deterministic_rate_rows.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["source"] == "65f25390f7d745d671818ba904c59b133ce1aa29"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    with np.load(path) as ref:
        assert ref["mvir"].shape == (3, 4)
        actual = Subhalos(2.0).Na_calc(ref["mvir"], ref["z"], 1e10, N_herm=1, sigmafac=0.0)
        np.testing.assert_array_equal(actual == 0, ref["expected"] == 0)
        directory = os.environ.get("SASHIMI_W_CORRECTED_REFERENCE_DIR")
        if directory is not None:
            report = json.loads((Path(directory) / "report.json").read_text())
            assert report["source"] == meta["source"]
            assert report["itamae_source"] == "23d01e8758a88b061b87de9e488c38ec89fd8e4f"
            runner = Path(directory) / "deterministic_rate_rows.npz"
            assert hashlib.sha256(runner.read_bytes()).hexdigest() == report["runner_files"]["deterministic_rate_rows"]
            with np.load(runner) as independent:
                assert set(independent.files) == {"mvir", "z", "expected"}
                np.testing.assert_array_equal(ref["mvir"], independent["mvir"])
                np.testing.assert_array_equal(ref["z"], independent["z"])
                np.testing.assert_array_equal(actual, independent["expected"])
        else:
            # A saved file is the fallback, not an additional cross-environment
            # gate in CI. CI uses exact independently installed source equality;
            # unchanged-source saved transport is reported separately.
            np.testing.assert_allclose(actual, ref["expected"], rtol=5e-12, atol=0.0)


def test_misaligned_leading_redshift_axis_fails_explicitly():
    with pytest.raises(ValueError, match="accretion-redshift"):
        Subhalos(2.0).Na_calc(np.ones((2, 4)), np.array([1.2, 1.5, 1.8]), 1e10, N_herm=1)
