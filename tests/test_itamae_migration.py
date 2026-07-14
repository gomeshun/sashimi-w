"""Regression tests for the SASHIMI-W ITAMAE migration boundary."""

import numpy as np

from itamae.units import AstropyUnits, NativeUnits
from itamae_migration import ItamaeSubhalos
import sashimi_w as legacy_module
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
    assert migrated.itamae_units.identifier == "native-units:1.0"


def test_wdm_variance_state_is_unchanged_by_growth_adapter() -> None:
    """The migration boundary must not modify sharp-k WDM variance arrays."""

    legacy = subhalos(mass_wdm=2.0)
    migrated = ItamaeSubhalos(mass_wdm=2.0)

    np.testing.assert_array_equal(migrated.filter_Mass, legacy.filter_Mass)
    np.testing.assert_allclose(migrated.Sigma, legacy.Sigma, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(migrated.Sigma_Sq, legacy.Sigma_Sq, rtol=0.0, atol=0.0)
    assert migrated.a == legacy.a


def _legacy_tuple() -> tuple[np.ndarray, ...]:
    """Return a small synthetic catalog in the documented legacy units."""

    size = 4
    return (
        np.array([1.0e6, 2.0e6, 3.0e6, 4.0e6]),
        np.array([0.5, 1.0, 1.5, 2.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.8e6, 1.5e6, 2.0e6, 3.5e6]),
        np.array([0.5, 1.0, 1.5, 2.0]),
        np.array([0.2, 0.4, 0.6, 0.8]),
        np.array([0.2, 0.8, 1.5, 3.0]),
        np.array([0.5, 1.0, 1.5, 2.0]),
        np.array([True, False, True, True]),
    )


def test_native_unit_backend_converts_legacy_catalog_to_canonical_units() -> None:
    """kpc and Msun/pc^3 output should become Mpc and Msun/Mpc^3."""

    migrated = ItamaeSubhalos(mass_wdm=2.0, unit_backend=NativeUnits())
    legacy = _legacy_tuple()
    catalog = migrated.catalog_from_legacy(legacy)

    np.testing.assert_array_equal(catalog.columns["m200_acc"], legacy[0])
    np.testing.assert_allclose(catalog.columns["r_s_acc"], legacy[2] * 1.0e-3)
    np.testing.assert_allclose(catalog.columns["rho_s_acc"], legacy[3] * 1.0e18)
    np.testing.assert_allclose(catalog.columns["r_s"], legacy[5] * 1.0e-3)
    np.testing.assert_allclose(catalog.columns["rho_s"], legacy[6] * 1.0e18)
    np.testing.assert_array_equal(
        catalog.weight_final, legacy[8] * legacy[9].astype(float)
    )
    assert set(catalog.weights) == {"weight_base", "weight_survival"}
    assert catalog.metadata["schema_version"] == "1.0"
    assert catalog.metadata["canonical_units"]["length"] == "Mpc"
    assert catalog.metadata["legacy_units"]["length"] == "kpc"


def test_astropy_and_native_unit_backends_produce_same_internal_catalog() -> None:
    """Unit-aware and lightweight paths should agree after canonicalization."""

    legacy = _legacy_tuple()
    native = ItamaeSubhalos(mass_wdm=2.0, unit_backend=NativeUnits())
    astropy = ItamaeSubhalos(mass_wdm=2.0, unit_backend=AstropyUnits())
    native_catalog = native.catalog_from_legacy(legacy)
    astropy_catalog = astropy.catalog_from_legacy(legacy)

    for name in native_catalog.columns:
        np.testing.assert_allclose(
            astropy_catalog.columns[name], native_catalog.columns[name], rtol=0.0, atol=0.0
        )
    np.testing.assert_array_equal(astropy_catalog.weight_final, native_catalog.weight_final)
    assert astropy_catalog.metadata["unit_backend"] == "astropy-units:1.0"


def test_exact_nfw_inverse_and_compatibility_aliases_are_temporary(monkeypatch) -> None:
    """Migration-only NumPy/SciPy shims and NFW replacement must be restored."""

    migrated = ItamaeSubhalos(mass_wdm=2.0)
    old_interp1d = legacy_module.interp1d
    had_alen = hasattr(np, "alen")
    old_alen = getattr(np, "alen", None)
    had_simps = hasattr(legacy_module.integrate, "simps")
    old_simps = getattr(legacy_module.integrate, "simps", None)
    expected = _legacy_tuple()

    def fake_legacy_calculation(self, *args, **kwargs):
        assert hasattr(np, "alen")
        assert hasattr(legacy_module.integrate, "simps")
        c_grid = np.linspace(0.0, 100.0, 1000)
        inverse = legacy_module.interp1d(
            self.fc(c_grid), c_grid, fill_value="extrapolate"
        )
        enclosed_fraction = np.array([1.0e-5, 0.01, 0.5, 2.0])
        concentration = inverse(enclosed_fraction)
        np.testing.assert_allclose(
            self.fc(concentration), enclosed_fraction, rtol=2.0e-11, atol=1.0e-13
        )
        return expected

    monkeypatch.setattr(subhalos, "rs_rhos_calc", fake_legacy_calculation)
    catalog = migrated.rs_rhos_catalog_calc(1.0e10)

    np.testing.assert_allclose(catalog.columns["r_s_acc"], expected[2] * 1.0e-3)
    assert legacy_module.interp1d is old_interp1d
    assert hasattr(np, "alen") is had_alen
    if had_alen:
        assert np.alen is old_alen
    assert hasattr(legacy_module.integrate, "simps") is had_simps
    if had_simps:
        assert legacy_module.integrate.simps is old_simps
