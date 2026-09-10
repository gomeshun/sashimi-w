"""The concentration inverse only evaluates its supported real branch."""

import numpy as np
import pytest

from sashimi_w import Subhalos, OmegaM, STANDARD_T2_Q10, Msolar


def test_nonreal_formation_trials_are_removed_before_growth():
    concentration = np.arange(1.0, 5.0)
    # At z=0 the real-branch boundary is density_ratio=1-OmegaM.
    density = np.array([0.0, 1 - OmegaM, 1.0, 2.0])
    c, ratio, z = Subhalos._real_formation_trials(concentration, density, 0.0)
    np.testing.assert_array_equal(c, [3.0, 4.0])
    np.testing.assert_array_equal(ratio, [1.0, 2.0])
    assert z[0] == pytest.approx(0.0)
    assert np.all(np.isfinite(z)) and np.all(z > -1)


def test_concentration_scalar_and_matrix_use_valid_background_inputs():
    model = Subhalos(2.0, wdm_power_convention=STANDARD_T2_Q10)
    scalar = model.conc200(1e9 * Msolar, 0.5)
    matrix = model.conc200(np.array([[1e9]]) * Msolar, np.array([[0.5]]))
    assert np.isfinite(scalar) and scalar > 0
    np.testing.assert_array_equal(matrix, [[scalar]])
