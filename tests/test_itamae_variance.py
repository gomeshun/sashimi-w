"""Regression tests for the SASHIMI-W variance protocol adapter."""

import numpy as np

from itamae_variance import make_variance_model
from sashimi_w import subhalos


def test_w_variance_adapter_matches_legacy_sharp_k_model() -> None:
    """The common interface should preserve WDM sigma and dS/dM arrays."""

    legacy = subhalos(mass_wdm=2.0)
    variance = make_variance_model(legacy)
    mass = np.array([1.0e7, 1.0e9, 1.0e11])
    redshift = np.array([0.0, 1.0, 3.0])

    np.testing.assert_allclose(
        variance.sigma(mass, redshift), legacy.sigmaMz(mass, redshift), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        variance.variance(mass, redshift),
        legacy.sigmaMz(mass, redshift) ** 2,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        variance.dvariance_dmass(mass, redshift),
        legacy.dsdm(mass, redshift),
        rtol=0.0,
        atol=0.0,
    )
    assert "m_wdm_keV=2.0" in variance.identifier
    assert "sharp-k" in variance.identifier
