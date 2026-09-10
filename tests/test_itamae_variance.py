"""Regression tests for the SASHIMI-W variance composition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from itamae.variance import (
    load_variance_cache,
    save_variance_cache,
    variance_cache_key,
)
from sashimi_w_itamae_migration import (
    ItamaeSubhalos,
    PUBLISHED_Q5,
    STANDARD_T2_Q10,
)
from sashimi_w_itamae_variance import (
    make_integrated_variance_model,
    make_variance_model,
)
from sashimi_w import h, k_file, sigma_8, subhalos

_GOLDEN = Path(__file__).parent / "golden" / "wdm_small_catalog.json"


def _fixture_mass_wdm() -> float:
    """Return the standard WDM particle mass recorded by the q5 fixture."""
    with _GOLDEN.open(encoding="utf-8") as input_file:
        return float(json.load(input_file)["parameters"]["mass_wdm"])


def test_callable_variance_adapter_exactly_preserves_legacy_model() -> None:
    """The reproduction adapter must retain native sigma and derivative arrays."""
    public = subhalos(mass_wdm=_fixture_mass_wdm())
    variance = make_variance_model(public)
    mass = np.array([1.0e7, 1.0e9, 1.0e11])
    redshift = np.array([0.0, 1.0, 3.0])

    np.testing.assert_array_equal(
        variance.sigma(mass, redshift),
        public.sigmaMz(mass, redshift),
    )
    np.testing.assert_allclose(
        variance.variance(mass, redshift),
        public.sigmaMz(mass, redshift) ** 2,
        rtol=2.0e-15,
    )
    np.testing.assert_array_equal(
        variance.dvariance_dmass(mass, redshift),
        public.dsdm(mass, redshift),
    )
    assert f"m_wdm_keV={_fixture_mass_wdm():g}" in variance.identifier
    assert "physics=legacy" in variance.identifier
    assert "power=published-q5" in variance.identifier
    assert "formula-role=q5-expression-used-as-power-ratio" in variance.identifier
    assert "q=5" in variance.identifier
    assert "half-mode=P_WDM/P_CDM=0.5" in variance.identifier
    assert "legacy-callable" in variance.identifier


def test_variance_factories_require_and_preserve_one_power_convention() -> None:
    """Factories cannot infer or override an ITAMAE power convention."""
    with pytest.raises(ValueError, match="wdm_power_convention"):
        make_variance_model()
    with pytest.raises(ValueError, match="wdm_power_convention"):
        make_integrated_variance_model()

    q5 = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    with pytest.raises(ValueError, match="does not match"):
        make_variance_model(q5, wdm_power_convention=STANDARD_T2_Q10)
    with pytest.raises(ValueError, match="does not match"):
        make_integrated_variance_model(
            q5,
            wdm_power_convention=STANDARD_T2_Q10,
        )


def test_callable_adapter_uses_consistent_derivative_when_requested() -> None:
    model = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    variance = make_variance_model(model)
    mass = np.array([1.0e8, 1.0e10])

    np.testing.assert_array_equal(
        variance.dvariance_dmass(mass, 2.0),
        model.dsdm(mass, 2.0),
    )
    assert "physics=consistent" in variance.identifier


def test_integrated_sharp_k_path_is_sigma8_normalized_and_continuous() -> None:
    """The corrected moving cutoff should produce finite nonzero derivatives."""
    model = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    variance = make_integrated_variance_model(model, n_k=1001)
    mass = np.geomspace(1.0e8, 1.0e14, 13)
    derivative = variance.dvariance_dmass(mass, 0.0)

    assert variance.sigma(model.MassIn8Mpc / h, 0.0) == pytest.approx(
        sigma_8,
        rel=2.0e-14,
    )
    assert variance.rho_mean == pytest.approx(model.Rhomean_z * h**2)
    assert variance.k_min == pytest.approx(k_file[0] * h)
    assert variance.k_max == pytest.approx(k_file[-1] * h)
    assert np.all(np.isfinite(derivative))
    assert np.all(derivative < 0.0)
    assert np.unique(derivative).size == derivative.size
    assert "integrated-variance:v3" in variance.identifier
    assert "filter_scale=6.0449698275617605" in variance.identifier
    assert "wdm-power-ratio" in variance.identifier
    assert "convention=published-q5" in variance.identifier
    assert "formula-role=q5-expression-used-as-power-ratio" in variance.identifier
    assert "q=5" in variance.identifier
    assert "source-k-unit=h/Mpc" in variance.identifier
    assert "unit-conversion=canonical-Mpc" in variance.identifier


def test_integrated_sigma_tracks_legacy_with_reviewed_quadrature_difference() -> None:
    """Accurate integration should remain close without reproducing rectangle bias."""
    model = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    variance = make_integrated_variance_model(model, n_k=2001)
    mass = np.geomspace(1.0e8, 1.0e14, 7)

    np.testing.assert_allclose(
        variance.sigma(mass, 0.0),
        model.Sigma_interp(h * mass),
        rtol=3.0e-2,
        atol=0.0,
    )
    np.testing.assert_allclose(
        variance.sigma(mass, 3.0),
        variance.sigma(mass, 0.0) * model.growthD(3.0),
        rtol=2.0e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        variance.dvariance_dmass(mass, 3.0),
        variance.dvariance_dmass(mass, 0.0) * model.growthD(3.0) ** 2,
        rtol=2.0e-14,
        atol=0.0,
    )


def test_standard_q10_integrated_variance_formula_and_identifier_are_pinned() -> None:
    """T-squared suppresses low-mass variance while retaining sigma8."""
    q5_model = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    q10_model = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="consistent",
        wdm_power_convention=STANDARD_T2_Q10,
    )
    q5 = make_integrated_variance_model(q5_model, n_k=1001)
    q10 = make_integrated_variance_model(q10_model, n_k=1001)
    sigma8_mass = q10_model.MassIn8Mpc / h

    assert q10.sigma(1.0e8, 0.0) < q5.sigma(1.0e8, 0.0)
    assert q10.sigma(sigma8_mass, 0.0) == pytest.approx(sigma_8, rel=2.0e-14)
    assert q5.sigma(sigma8_mass, 0.0) == pytest.approx(sigma_8, rel=2.0e-14)
    assert q10.identifier != q5.identifier
    assert "convention=standard-t2-q10" in q10.identifier
    assert "formula-role=viel-transfer-amplitude-squared" in q10.identifier
    assert "q=10" in q10.identifier
    assert "half-mode=T_WDM/T_CDM=0.5;P_WDM/P_CDM=0.25" in q10.identifier

    callable_q10 = make_variance_model(q10_model)
    assert "power=standard-t2-q10" in callable_q10.identifier
    assert "formula-role=viel-transfer-amplitude-squared" in callable_q10.identifier
    assert "q=10" in callable_q10.identifier
    assert "half-mode=T_WDM/T_CDM=0.5;P_WDM/P_CDM=0.25" in (callable_q10.identifier)


@pytest.mark.parametrize("convention", [PUBLISHED_Q5, STANDARD_T2_Q10])
def test_variance_and_derivative_use_one_continuous_integral(convention) -> None:
    """Mass interpolation/projection must not decouple sigma from its slope."""
    model = ItamaeSubhalos(mass_wdm=2.0, wdm_power_convention=convention)
    mass = np.geomspace(1.0e9, 1.0e13, 24)
    step = 1.0e-4
    upper, lower = mass * np.exp(step), mass * np.exp(-step)
    numerical = (model.sigmaMz(upper, 0.5) ** 2 - model.sigmaMz(lower, 0.5) ** 2) / (upper - lower)
    np.testing.assert_allclose(model.dsdm(mass, 0.5), numerical, rtol=4e-8, atol=0.0)
    np.testing.assert_array_equal(
        model.sigmaMz(mass, 0.5), model._consistent_variance().sigma(mass, 0.5)
    )
    # The model composition is reused; no projected mass-variance table exists.
    variance = model._consistent_variance()
    model.sigmaMz(mass[::-1], 1.0)
    assert model._consistent_variance() is variance
    assert not hasattr(model, "_consistent_sigma_z0_values")


def test_lighter_wdm_has_more_small_scale_power_suppression() -> None:
    light = make_integrated_variance_model(
        mass_wdm=1.0,
        wdm_power_convention=PUBLISHED_Q5,
        n_k=1001,
    )
    heavy = make_integrated_variance_model(
        mass_wdm=5.0,
        wdm_power_convention=PUBLISHED_Q5,
        n_k=1001,
    )

    assert light.sigma(1.0e8, 0.0) < heavy.sigma(1.0e8, 0.0)
    assert light.sigma(1.0e14, 0.0) == pytest.approx(
        heavy.sigma(1.0e14, 0.0),
        rel=3.0e-3,
    )


def test_integrated_path_rejects_legacy_derivative_convention() -> None:
    model = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="legacy",
        wdm_power_convention=PUBLISHED_Q5,
    )
    with pytest.raises(ValueError, match="consistent"):
        make_integrated_variance_model(model)


def test_variance_cache_binds_wdm_particle_mass_and_round_trips(tmp_path) -> None:
    model = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="consistent",
        wdm_power_convention=PUBLISHED_Q5,
    )
    variance_model = make_integrated_variance_model(model, n_k=501)
    mass = np.geomspace(1.0e8, 1.0e12, 5)
    variance = variance_model.variance(mass)
    settings = {
        "physics_mode": "consistent",
        "mass_wdm_keV": model.mass_wdm,
        "wdm_power_convention": model.wdm_power_convention,
        "wdm_power_formula_role": model.wdm_power_formula_role,
        "wdm_power_q": model.wdm_power_q,
        "half_mode_definition": model.half_mode_definition,
    }
    key = variance_cache_key(
        variance_model.identifier,
        mass,
        backend_identifier=model.itamae_cosmology.identifier,
        settings=settings,
    )
    changed = variance_cache_key(
        variance_model.identifier,
        mass,
        backend_identifier=model.itamae_cosmology.identifier,
        settings={**settings, "mass_wdm_keV": 3.0},
    )
    assert key != changed
    q10_model = ItamaeSubhalos(
        mass_wdm=_fixture_mass_wdm(),
        physics_mode="consistent",
        wdm_power_convention=STANDARD_T2_Q10,
    )
    q10_variance_model = make_integrated_variance_model(q10_model, n_k=501)
    q10_settings = {
        "physics_mode": "consistent",
        "mass_wdm_keV": q10_model.mass_wdm,
        "wdm_power_convention": q10_model.wdm_power_convention,
        "wdm_power_formula_role": q10_model.wdm_power_formula_role,
        "wdm_power_q": q10_model.wdm_power_q,
        "half_mode_definition": q10_model.half_mode_definition,
    }
    q10_key = variance_cache_key(
        q10_variance_model.identifier,
        mass,
        backend_identifier=q10_model.itamae_cosmology.identifier,
        settings=q10_settings,
    )
    assert key != q10_key

    cache = tmp_path / "variance.npz"
    save_variance_cache(cache, key=key, mass=mass, variance=variance)
    loaded_mass, loaded_variance = load_variance_cache(
        cache,
        expected_key=key,
    )
    np.testing.assert_array_equal(loaded_mass, mass)
    np.testing.assert_array_equal(loaded_variance, variance)
    with pytest.raises(ValueError, match="key"):
        load_variance_cache(cache, expected_key=changed)
    with pytest.raises(ValueError, match="key"):
        load_variance_cache(cache, expected_key=q10_key)
