"""Regression tests for the SASHIMI-W ITAMAE migration boundary."""

import numpy as np

from itamae_migration import ItamaeSubhalos
from sashimi_w import OmegaM, h, subhalos


def test_itamae_growth_matches_legacy_redshift_modes() -> None:
    """Scalar, vector, and interval growth conventions should be preserved."""

    legacy = subhalos(mass_wdm=2.0)
    migrated = ItamaeSubhalos(mass_wdm=2.0)
    omega_lambda = 1.0 - OmegaM

    for redshift in (0.0, 0.5, 3.0, np.array([0.0, 0.5, 1.0, 3.0])):
        np.testing.assert_allclose(
            migrated.linear_growth_factor(OmegaM, omega_lambda, redshift),
            legacy.linear_growth_factor(OmegaM, omega_lambda, redshift),
            rtol=2.0e-12,
            atol=0.0,
        )

    interval = np.array([0.5, 3.0])
    np.testing.assert_allclose(
        migrated.linear_growth_factor(OmegaM, omega_lambda, interval),
        legacy.linear_growth_factor(OmegaM, omega_lambda, interval),
        rtol=2.0e-12,
        atol=0.0,
    )


def test_growth_backend_uses_wmap7_configuration() -> None:
    """The default ITAMAE backend should retain the WMAP7 module parameters."""

    migrated = ItamaeSubhalos(mass_wdm=2.0)

    assert np.isclose(migrated.itamae_cosmology.omega_m0, OmegaM)
    assert np.isclose(migrated.itamae_cosmology.h, h)


def test_wdm_variance_state_is_unchanged_by_growth_adapter() -> None:
    """The first migration step must not modify sharp-k WDM variance arrays."""

    legacy = subhalos(mass_wdm=2.0)
    migrated = ItamaeSubhalos(mass_wdm=2.0)

    np.testing.assert_array_equal(migrated.filter_Mass, legacy.filter_Mass)
    np.testing.assert_allclose(migrated.Sigma, legacy.Sigma, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(migrated.Sigma_Sq, legacy.Sigma_Sq, rtol=0.0, atol=0.0)
    assert migrated.a == legacy.a
