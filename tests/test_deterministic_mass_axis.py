"""The one-history branch must distinguish redshift and mass axes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sashimi_w import Subhalos


def test_nonsquare_grid_matches_independent_prechange_single_rows():
    path = Path(__file__).parent / "references/deterministic_rate_rows.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["source"] == "dcef1910d42cab940567be448ffc79b42d436802"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    with np.load(path) as ref:
        assert ref["mvir"].shape == (3, 4)
        actual = Subhalos(2.0).Na_calc(ref["mvir"], ref["z"], 1e10, N_herm=1, sigmafac=0.0)
        np.testing.assert_allclose(actual, ref["expected"], rtol=5e-12, atol=0.0)


def test_misaligned_leading_redshift_axis_fails_explicitly():
    with pytest.raises(ValueError, match="accretion-redshift"):
        Subhalos(2.0).Na_calc(np.ones((2, 4)), np.array([1.2, 1.5, 1.8]), 1e10, N_herm=1)
