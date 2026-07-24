"""Physics and public-boundary regressions for the SASHIMI-W migration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from itamae.cosmology import NativeFlatLCDM
from itamae.types import WeightedSubhaloCatalog
from itamae.units import AstropyUnits, NativeUnits
from sashimi_w_itamae_migration import ItamaeSubhalos, PUBLISHED_Q5, STANDARD_T2_Q10
import sashimi_w as legacy_module
from sashimi_w import OmegaM, h, subhalos
import sashimi_w_itamae


_GOLDEN = Path(__file__).parent / "golden" / "wdm_small_catalog.json"
_Q10_GOLDEN = Path(__file__).parent / "golden" / "wdm_q10_small_catalog.json"
_COLUMN_MAPPING = {
    "m200_acc": ("m200_acc", 1.0),
    "z_acc": ("z_acc", 1.0),
    "r_s_acc": ("r_s_acc_kpc", 1.0e-3),
    "rho_s_acc": ("rho_s_acc_msun_pc3", 1.0e18),
    "m_bound": ("m_bound", 1.0),
    "r_s": ("r_s_kpc", 1.0e-3),
    "rho_s": ("rho_s_msun_pc3", 1.0e18),
    "c_t": ("c_t", 1.0),
}


def _fixture() -> dict:
    return json.loads(_GOLDEN.read_text())


def _q10_fixture() -> dict:
    return json.loads(_Q10_GOLDEN.read_text())


def _synthetic_legacy_tuple() -> tuple[np.ndarray, ...]:
    """Return a compact catalog in the historical implicit units."""
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


def test_opt_in_facade_does_not_replace_legacy_public_api() -> None:
    """The established import path must remain the historical class."""
    assert legacy_module.subhalos is subhalos
    assert sashimi_w_itamae.subhalos is ItamaeSubhalos
    assert not issubclass(subhalos, ItamaeSubhalos)


def test_wdm_power_convention_is_explicit_and_validated() -> None:
    """New migrated callers cannot silently inherit a power convention."""
    with pytest.raises(TypeError, match="wdm_power_convention"):
        ItamaeSubhalos()
    with pytest.raises(ValueError, match="wdm_power_convention"):
        ItamaeSubhalos(wdm_power_convention="mixed-q")


def test_q5_q10_formulas_half_mode_and_consumers_are_coherent() -> None:
    """One explicit convention must feed both EPS and concentration power."""
    public = subhalos(mass_wdm=2.0)
    q5 = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    q10 = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="consistent",
        wdm_power_convention=STANDARD_T2_Q10,
    )
    q10_legacy_growth = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="legacy",
        wdm_power_convention=STANDARD_T2_Q10,
    )
    wavenumber = np.geomspace(0.05, 100.0, 11)
    published_ratio = q5._wdm_power_ratio(wavenumber)

    np.testing.assert_array_equal(
        published_ratio,
        public._wdm_power_ratio(wavenumber),
    )
    np.testing.assert_allclose(
        q10._wdm_power_ratio(wavenumber),
        published_ratio**2,
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        q10_legacy_growth._wdm_power_ratio(wavenumber),
        q10._wdm_power_ratio(wavenumber),
    )

    half_mode = q5.half_mode_wavenumber()
    assert q10.half_mode_wavenumber() == pytest.approx(half_mode, rel=0.0, abs=0.0)
    assert q5._wdm_power_ratio(half_mode) == pytest.approx(0.5, rel=3.0e-15)
    assert q10._wdm_power_ratio(half_mode) == pytest.approx(0.25, rel=3.0e-15)
    assert q5.half_mode_definition == "P_WDM/P_CDM=0.5"
    assert q10.half_mode_definition == "T_WDM/T_CDM=0.5;P_WDM/P_CDM=0.25"

    # SigmaIntegrand is the top-hat power path used by the Ludlow
    # concentration calibration. The inherited initialization independently
    # constructs the sharp-k Sigma interpolation used by EPS.
    concentration_k = np.geomspace(0.05, 20.0, 9)
    np.testing.assert_allclose(
        q10.SigmaIntegrand(concentration_k, 0.05),
        q5.SigmaIntegrand(concentration_k, 0.05)
        * q5._wdm_power_ratio(concentration_k),
        rtol=4.0e-15,
        atol=0.0,
    )
    assert q10.Sigma_interp(1.0e8) < q5.Sigma_interp(1.0e8)
    assert q10.conc200(1.0e8 * legacy_module.Msolar, 1.0) < q5.conc200(
        1.0e8 * legacy_module.Msolar,
        1.0,
    )


def test_growth_and_derivative_modes_are_explicit_and_self_consistent() -> None:
    """Legacy reproduction and corrected flat-WMAP7 evolution must stay separate."""
    public = subhalos(mass_wdm=2.0)
    legacy = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="legacy",
        wdm_power_convention=PUBLISHED_Q5,
    )
    consistent = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    redshift = np.array([0.0, 0.5, 1.0, 3.0])
    mass = np.array([1.0e8, 1.0e10, 1.0e12])

    np.testing.assert_allclose(legacy.growthD(redshift), public.growthD(redshift))
    np.testing.assert_allclose(legacy.dDdz(redshift), public.dDdz(redshift))
    np.testing.assert_allclose(legacy.dsdm(mass, 1.0), public.dsdm(mass, 1.0))
    assert legacy.omega_lambda == pytest.approx(0.7769)
    assert legacy.growthD(0.0) == pytest.approx(0.9375427442633785)

    assert consistent.omega_lambda == pytest.approx(1.0 - OmegaM)
    assert consistent.growthD(0.0) == pytest.approx(1.0)
    assert consistent.Hz(0.0) == pytest.approx(legacy_module.H0)
    for value in redshift:
        step = 1.0e-5 * (1.0 + value)
        numerical = (consistent.growthD(value + step) - consistent.growthD(value - step)) / (
            2.0 * step
        )
        assert consistent.dDdz(value) == pytest.approx(numerical, rel=2.0e-9)

    np.testing.assert_allclose(
        consistent.dsdm(mass, 1.0),
        consistent._consistent_variance().dvariance_dmass(mass, 1.0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        consistent.sigmaMz(mass, 1.0),
        consistent._consistent_variance().sigma(mass, 1.0),
        rtol=3.5e-3,
        atol=0.0,
    )


def test_signed_legacy_variance_is_reproduced_but_not_mislabeled_as_counts() -> None:
    """Legacy signed weights stay in tuples; consistent catalogs remove their cause."""

    public = subhalos(mass_wdm=2.0)
    legacy = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="legacy",
        wdm_power_convention=PUBLISHED_Q5,
    )
    consistent = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    mass = np.geomspace(1.0e5, 1.0e8, 128)

    public_derivative = public.dsdm(mass, 0.0)
    np.testing.assert_array_equal(legacy.dsdm(mass, 0.0), public_derivative)
    assert np.any(public_derivative > 0.0)
    assert np.all(consistent.dsdm(mass, 0.0) < 0.0)

    signed_tuple = list(_synthetic_legacy_tuple())
    signed_tuple[8] = np.array([0.5, -0.1, 1.5, 2.0])
    with pytest.raises(
        ValueError,
        match=r"signed population weights.*not clipped or renormalized",
    ):
        legacy.catalog_from_legacy(tuple(signed_tuple))

    # The compatibility tuple remains untouched for exact reproduction.
    np.testing.assert_array_equal(signed_tuple[8], np.array([0.5, -0.1, 1.5, 2.0]))


def test_consistent_accretion_rate_uses_each_redshift_mass_grid() -> None:
    """Only corrected mode should stop reusing the final redshift mass row."""

    legacy = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="legacy",
        wdm_power_convention=PUBLISHED_Q5,
    )
    consistent = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    mass_by_redshift = np.array(
        [
            [1.1e6, 1.1e7, 1.1e8],
            [1.2e6, 1.2e7, 1.2e8],
        ]
    )
    final_mass = mass_by_redshift[-1]

    np.testing.assert_array_equal(
        legacy._select_accretion_mass_grid(mass_by_redshift, final_mass),
        final_mass,
    )
    np.testing.assert_array_equal(
        consistent._select_accretion_mass_grid(mass_by_redshift, final_mass),
        mass_by_redshift,
    )


def test_concentration_boundary_converts_physical_mass_to_msun_per_h() -> None:
    """Consistent concentration must evaluate the legacy grid at h*M."""
    public = subhalos(mass_wdm=2.0)
    legacy = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="legacy",
        wdm_power_convention=PUBLISHED_Q5,
    )
    consistent = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    mass_cgs = np.array([1.0e8, 1.0e10, 1.0e12]) * legacy_module.Msolar
    redshift = 1.0

    np.testing.assert_allclose(
        legacy.conc200(mass_cgs, redshift),
        public.conc200(mass_cgs, redshift),
        rtol=1.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        consistent.conc200(mass_cgs, redshift),
        public.conc200(mass_cgs * h**2, redshift),
        rtol=1.0e-15,
        atol=0.0,
    )
    assert not np.allclose(
        consistent.conc200(mass_cgs, redshift),
        public.conc200(mass_cgs, redshift),
        rtol=1.0e-6,
        atol=0.0,
    )


def test_itamae_growth_interval_api_retains_the_established_convention() -> None:
    """The separate interval helper remains compatible with the original API."""
    public = subhalos(mass_wdm=2.0)
    migrated = ItamaeSubhalos(
        mass_wdm=2.0,
        wdm_power_convention=PUBLISHED_Q5,
    )
    omega_lambda = 1.0 - OmegaM

    for redshift in (0.0, 0.5, 3.0, np.array([0.0, 0.5, 1.0, 3.0])):
        np.testing.assert_allclose(
            migrated.linear_growth_factor(OmegaM, omega_lambda, redshift),
            public.linear_growth_factor(OmegaM, omega_lambda, redshift),
            rtol=2.0e-12,
        )
    np.testing.assert_allclose(
        migrated.linear_growth_factor(
            OmegaM,
            omega_lambda,
            np.array([0.5, 3.0]),
        ),
        public.linear_growth_factor(
            OmegaM,
            omega_lambda,
            np.array([0.5, 3.0]),
        ),
        rtol=2.0e-12,
    )


def test_native_and_astropy_units_produce_the_same_canonical_catalog() -> None:
    """Implicit kpc and Msun/pc3 values must cross one explicit unit boundary."""
    legacy = _synthetic_legacy_tuple()
    native = ItamaeSubhalos(
        mass_wdm=2.0,
        unit_backend=NativeUnits(),
        wdm_power_convention=PUBLISHED_Q5,
    )
    astropy = ItamaeSubhalos(
        mass_wdm=2.0,
        unit_backend=AstropyUnits(),
        wdm_power_convention=PUBLISHED_Q5,
    )
    native_catalog = native.catalog_from_legacy(legacy)
    astropy_catalog = astropy.catalog_from_legacy(legacy)

    np.testing.assert_array_equal(native_catalog.columns["m200_acc"], legacy[0])
    np.testing.assert_allclose(native_catalog.columns["r_s_acc"], legacy[2] * 1.0e-3)
    np.testing.assert_allclose(
        native_catalog.columns["rho_s_acc"],
        legacy[3] * 1.0e18,
    )
    assert set(native_catalog.weights) == {
        "weight_base",
        "weight_concentration",
        "weight_survival",
    }
    np.testing.assert_array_equal(
        native_catalog.weight_final,
        legacy[8] * legacy[9].astype(float),
    )
    for name in native_catalog.columns:
        np.testing.assert_allclose(
            astropy_catalog.columns[name],
            native_catalog.columns[name],
            rtol=0.0,
            atol=0.0,
        )
    assert astropy_catalog.metadata["unit_backend"] == "astropy-units:1.0"


def test_nfw_inverse_is_injected_without_mutating_module_globals() -> None:
    """Consistent mode uses ITAMAE exactly while legacy mode stays reproducible."""
    legacy = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="legacy",
        wdm_power_convention=PUBLISHED_Q5,
    )
    consistent = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    old_interp = legacy_module.interp1d
    old_simpson = legacy_module.integrate.simpson
    fraction = np.array([1.0e-5, 0.01, 0.5, 2.0])

    legacy_concentration = legacy._invert_nfw_mass_fraction(fraction)
    public_concentration = subhalos(mass_wdm=2.0)._invert_nfw_mass_fraction(fraction)
    np.testing.assert_array_equal(legacy_concentration, public_concentration)

    exact_concentration = consistent._invert_nfw_mass_fraction(fraction)
    np.testing.assert_allclose(
        consistent.fc(exact_concentration),
        fraction,
        rtol=2.0e-11,
        atol=1.0e-13,
    )
    assert legacy_module.interp1d is old_interp
    assert legacy_module.integrate.simpson is old_simpson
    assert not hasattr(np, "alen")


@pytest.mark.parametrize("physics_mode", ["legacy", "consistent"])
def test_full_catalog_matches_mode_specific_golden_and_invariants(
    physics_mode: str,
    tmp_path: Path,
) -> None:
    """Every compact full-catalog field is pinned for both physics modes."""
    fixture = _fixture()
    expected = fixture["modes"][physics_mode]
    assert fixture["units"]["catalog_mass"] == "physical Msun"
    model = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode=physics_mode,
        wdm_power_convention=PUBLISHED_Q5,
    )
    catalog = model.rs_rhos_catalog_calc(**fixture["parameters"])

    assert catalog.shape == (16,)
    for column, (golden_name, scale) in _COLUMN_MAPPING.items():
        np.testing.assert_allclose(
            catalog.columns[column],
            np.asarray(expected[golden_name]) * scale,
            rtol=5.0e-10,
            atol=1.0e-14,
        )
    survive = np.asarray(expected["survive"], dtype=bool)
    weight = np.asarray(expected["weight"])
    np.testing.assert_array_equal(catalog.columns["survive"], survive)
    np.testing.assert_allclose(
        catalog.weights["weight_base"] * catalog.weights["weight_concentration"],
        weight,
        rtol=5.0e-12,
        atol=1.0e-18,
    )
    np.testing.assert_allclose(
        catalog.weight_final,
        weight * survive,
        rtol=5.0e-12,
        atol=1.0e-18,
    )
    assert np.all(catalog.columns["m_bound"] <= catalog.columns["m200_acc"])
    np.testing.assert_array_equal(
        catalog.columns["survive"],
        catalog.columns["c_t"] > 0.77,
    )

    reconstructed_fraction = catalog.columns["m_bound"] / (
        4.0
        * np.pi
        * (catalog.columns["rho_s"] / 1.0e18)
        * (catalog.columns["r_s"] * 1.0e3 * 1.0e3) ** 3
    )
    # Legacy interpolation used only 1000 concentration nodes, so its inverse
    # reconstructs enclosed mass to about 1e-4. Consistent mode uses ITAMAE's
    # bracketed inverse and closes the identity to floating-point precision.
    reconstruction_tolerance = 1.5e-4 if physics_mode == "legacy" else 3.0e-10
    np.testing.assert_allclose(
        model.fc(catalog.columns["c_t"]),
        reconstructed_fraction,
        rtol=reconstruction_tolerance,
        atol=2.0e-13,
    )
    assert catalog.metadata["physics_mode"] == physics_mode
    assert catalog.metadata["wdm_power_convention"] == PUBLISHED_Q5
    assert catalog.metadata["wdm_power_q"] == 5.0
    assert catalog.metadata["variance_growth_power"] == (2 if physics_mode == "consistent" else 1)
    if physics_mode == "consistent":
        assert catalog.metadata["model_identifier"].endswith("itamae-migration:v5")
        assert (
            catalog.metadata["accretion_mass_redshift_mapping"]
            == "per-redshift-virial-mass-grid"
        )
        assert catalog.metadata["population_weight_contract"] == (
            "nonnegative-corrected-counts"
        )
        assert fixture["units"]["consistent_variance_mass"] == "physical Msun"
        assert catalog.metadata["variance_mass_unit"] == "Msun"
        assert catalog.metadata["variance_power_units"] == {
            "wavenumber": "1/Mpc",
            "power": "Mpc^3",
            "density": "Msun/Mpc^3",
        }
        assert catalog.metadata["physical_to_filter_mass"] == (
            "M_filter[Msun/h] = h * M_physical[Msun]"
        )
    else:
        assert catalog.metadata["model_identifier"].endswith("itamae-migration:v4")
        assert catalog.metadata["accretion_mass_redshift_mapping"] == (
            "legacy-final-redshift-grid-reused"
        )
        assert catalog.metadata["population_weight_contract"].startswith(
            "exact-signed-tuple"
        )
        assert "Msun/h grid" in fixture["units"]["legacy_variance_mass"]
        assert "legacy-raw" in catalog.metadata["variance_mass_unit"]
    assert catalog.metadata["itamae_version"] == "0.1.0a4"

    archive = tmp_path / f"{physics_mode}.npz"
    catalog.to_npz(archive)
    restored = WeightedSubhaloCatalog.from_npz(archive)
    for name in catalog.columns:
        np.testing.assert_array_equal(restored.columns[name], catalog.columns[name])
    np.testing.assert_array_equal(restored.weight_final, catalog.weight_final)
    assert dict(restored.metadata) == dict(catalog.metadata)


def test_legacy_full_catalog_reproduces_the_public_tuple() -> None:
    """Compatibility mode should preserve all historical tuple fields."""
    parameters = _fixture()["parameters"]
    public = subhalos(mass_wdm=2.0).rs_rhos_calc(**parameters)
    migrated_model = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="legacy",
        wdm_power_convention=PUBLISHED_Q5,
    )
    migrated = migrated_model.rs_rhos_calc(**parameters)

    for index, (actual, expected) in enumerate(zip(migrated, public, strict=True)):
        if index == 9:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=2.0e-9,
                atol=1.0e-13,
            )

    catalog = migrated_model.rs_rhos_catalog_calc(**parameters)
    public_columns = {
        "m200_acc": (public[0], 1.0),
        "z_acc": (public[1], 1.0),
        "r_s_acc": (public[2], 1.0e-3),
        "rho_s_acc": (public[3], 1.0e18),
        "m_bound": (public[4], 1.0),
        "r_s": (public[5], 1.0e-3),
        "rho_s": (public[6], 1.0e18),
        "c_t": (public[7], 1.0),
    }
    for name, (values, scale) in public_columns.items():
        np.testing.assert_allclose(
            catalog.columns[name],
            np.asarray(values) * scale,
            rtol=2.0e-9,
            atol=1.0e-13,
        )
    np.testing.assert_array_equal(catalog.columns["survive"], public[9])
    np.testing.assert_allclose(
        catalog.weights["weight_base"] * catalog.weights["weight_concentration"],
        public[8],
        rtol=5.0e-12,
        atol=1.0e-100,
    )
    np.testing.assert_allclose(
        catalog.weight_final,
        public[8] * public[9],
        rtol=5.0e-12,
        atol=1.0e-100,
    )


def test_legacy_mass_function_and_cumulative_satellites_reproduce_public_api() -> None:
    """Derived legacy observables must agree, not only the underlying tuple."""
    parameters = dict(_fixture()["parameters"])
    host_mass = parameters.pop("M0")
    public = subhalos(mass_wdm=2.0)
    migrated = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode="legacy",
        wdm_power_convention=PUBLISHED_Q5,
    )

    public_mass, public_dndm = public.subhalo_distr(host_mass, **parameters)
    migrated_mass, migrated_dndm = migrated.subhalo_distr(host_mass, **parameters)
    np.testing.assert_allclose(
        migrated_mass,
        public_mass,
        rtol=2.0e-9,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        migrated_dndm,
        public_dndm,
        rtol=2.0e-9,
        atol=1.0e-100,
    )

    public_total, public_threshold, public_cumulative = public.N_sat(
        host_mass,
        **parameters,
    )
    migrated_total, migrated_threshold, migrated_cumulative = migrated.N_sat(
        host_mass,
        **parameters,
    )
    assert migrated_total == pytest.approx(public_total, rel=2.0e-9, abs=1.0e-100)
    np.testing.assert_allclose(
        migrated_threshold,
        public_threshold,
        rtol=2.0e-9,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        migrated_cumulative,
        public_cumulative,
        rtol=2.0e-9,
        atol=1.0e-100,
    )


def test_standard_q10_full_catalog_golden_metadata_and_roundtrip(
    tmp_path: Path,
) -> None:
    """The standard T-squared path is pinned as a complete catalog artifact."""
    fixture = _q10_fixture()
    assert fixture["wdm_power_convention"] == STANDARD_T2_Q10
    assert fixture["physics_mode"] == "consistent"
    model = ItamaeSubhalos(
        mass_wdm=2.0,
        physics_mode=fixture["physics_mode"],
        wdm_power_convention=fixture["wdm_power_convention"],
    )
    catalog = model.rs_rhos_catalog_calc(**fixture["parameters"])
    expected = fixture["catalog"]

    assert catalog.shape == (16,)
    for column, (golden_name, scale) in _COLUMN_MAPPING.items():
        np.testing.assert_allclose(
            catalog.columns[column],
            np.asarray(expected[golden_name]) * scale,
            rtol=5.0e-10,
            atol=1.0e-14,
        )
    survive = np.asarray(expected["survive"], dtype=bool)
    weight = np.asarray(expected["weight"])
    np.testing.assert_array_equal(catalog.columns["survive"], survive)
    np.testing.assert_allclose(
        catalog.weights["weight_base"] * catalog.weights["weight_concentration"],
        weight,
        rtol=5.0e-12,
        atol=1.0e-100,
    )
    np.testing.assert_allclose(
        catalog.weight_final,
        weight * survive,
        rtol=5.0e-12,
        atol=1.0e-100,
    )

    metadata = catalog.metadata
    assert metadata["wdm_power_convention"] == STANDARD_T2_Q10
    assert metadata["wdm_power_formula_role"] == "viel-transfer-amplitude-squared"
    assert metadata["wdm_power_q"] == 10.0
    assert metadata["half_mode_definition"] == (
        "T_WDM/T_CDM=0.5;P_WDM/P_CDM=0.25"
    )
    assert metadata["half_mode_power_ratio"] == 0.25
    assert metadata["half_mode_wavenumber_h_per_mpc"] == pytest.approx(
        model.half_mode_wavenumber(),
    )
    assert "power=standard-t2-q10" in metadata["model_identifier"]
    assert metadata["model_identifier"].endswith("itamae-migration:v5")
    assert metadata["accretion_mass_redshift_mapping"] == (
        "per-redshift-virial-mass-grid"
    )

    archive = tmp_path / "standard-t2-q10.npz"
    catalog.to_npz(archive)
    restored = WeightedSubhaloCatalog.from_npz(archive)
    for name in catalog.columns:
        np.testing.assert_array_equal(restored.columns[name], catalog.columns[name])
    np.testing.assert_array_equal(restored.weight_final, catalog.weight_final)
    assert dict(restored.metadata) == dict(catalog.metadata)


def test_consistent_mode_records_a_material_abundance_change() -> None:
    """The reviewed coupled corrections must not collapse back to legacy output."""
    fixture = _fixture()
    legacy_total = np.sum(fixture["modes"]["legacy"]["weight"])
    consistent_total = np.sum(fixture["modes"]["consistent"]["weight"])

    assert consistent_total / legacy_total == pytest.approx(
        0.18222963347293586,
        rel=2.0e-14,
    )


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"mass_wdm": 0.0}, ValueError, "mass_wdm"),
        ({"mass_wdm": np.nan}, ValueError, "mass_wdm"),
        ({"physics_mode": "mixed"}, ValueError, "physics_mode"),
        (
            {
                "cosmology_backend": NativeFlatLCDM(
                    omega_m0=0.3,
                    h=h,
                )
            },
            ValueError,
            "OmegaM",
        ),
    ],
)
def test_constructor_rejects_mixed_or_invalid_physics(
    arguments: dict,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ItamaeSubhalos(
            wdm_power_convention=PUBLISHED_Q5,
            **arguments,
        )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"dz": 0.0}, ValueError, "dz"),
        ({"zmax": 0.0}, ValueError, "zmax"),
        ({"N_ma": 1}, ValueError, "N_ma"),
        ({"N_herm": 0}, ValueError, "N_herm"),
        ({"logmamin": 9.0, "logmamax": 8.0}, ValueError, "logmamin"),
        ({"profile_change": "yes"}, TypeError, "profile_change"),
    ],
)
def test_catalog_input_validation_fails_before_numerical_work(
    overrides: dict,
    error: type[Exception],
    message: str,
) -> None:
    parameters = _fixture()["parameters"] | overrides
    with pytest.raises(error, match=message):
        ItamaeSubhalos(
            wdm_power_convention=PUBLISHED_Q5,
        ).rs_rhos_catalog_calc(**parameters)


def test_factorized_weight_mismatch_is_rejected() -> None:
    model = ItamaeSubhalos(wdm_power_convention=PUBLISHED_Q5)
    legacy = _synthetic_legacy_tuple()
    with pytest.raises(ValueError, match="reconstruct"):
        model.catalog_from_legacy(
            legacy,
            weight_base=np.ones(4),
            weight_concentration=np.ones(4),
        )
