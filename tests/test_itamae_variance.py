"""Regression tests for the SASHIMI-W variance composition."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from itamae.variance import load_variance_cache, save_variance_cache, variance_cache_key
from sashimi_w_itamae_migration import ItamaeSubhalos, STANDARD_T2_Q10
from sashimi_w_itamae_variance import make_integrated_variance_model, make_variance_model
from sashimi_w import h, k_file, Pk_file, sigma_8, subhalos

_GOLDEN = Path(__file__).parent / "golden" / "wdm_q10_small_catalog.json"


def _fixture_mass_wdm() -> float:
    """Return the standard WDM particle mass recorded by the q10 fixture.

    Returns
    -------
    float
        WDM particle mass in keV recorded by the frozen q10 fixture.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    with _GOLDEN.open(encoding="utf-8") as input_file:
        return float(json.load(input_file)["parameters"]["mass_wdm"])


def test_callable_variance_adapter_preserves_standard_model() -> None:
    """Preserve the WDM variance and derivative through the callable ITAMAE interface.

    Notes
    -----
    Pass criteria: Sigma and derivative equal the standard model, variance equals sigma
    squared, and metadata distinguish particle mass, q10 and half-mode convention.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    public = subhalos(mass_wdm=_fixture_mass_wdm(), wdm_power_convention=STANDARD_T2_Q10)
    variance = make_variance_model(public)
    mass = np.array([10000000.0, 1000000000.0, 100000000000.0])
    redshift = np.array([0.0, 1.0, 3.0])
    np.testing.assert_array_equal(variance.sigma(mass, redshift), public.sigmaMz(mass, redshift))
    np.testing.assert_allclose(
        variance.variance(mass, redshift), public.sigmaMz(mass, redshift) ** 2, rtol=2e-15
    )
    np.testing.assert_array_equal(
        variance.dvariance_dmass(mass, redshift), public.dsdm(mass, redshift)
    )
    assert f"m_wdm_keV={_fixture_mass_wdm():g}" in variance.identifier
    assert "convention=standard-t2-q10" in variance.identifier
    assert "formula-role=viel-transfer-amplitude-squared" in variance.identifier
    assert "q=10" in variance.identifier
    assert "half-mode=T_WDM/T_CDM=0.5;P_WDM/P_CDM=0.25" in variance.identifier


def test_variance_factories_default_to_q10_and_reject_old_models() -> None:
    r"""Prevent either variance factory from silently accepting the retired q5 prescription.

    Notes
    -----
    Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
    nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
    Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

    Pass criteria: Both factories identify q10 defaults and reject a supplied q5-like model.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.

    References
    ----------

    .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
        https://arxiv.org/abs/astro-ph/0501562
    """
    assert "q=10" in make_variance_model().identifier
    assert "q=10" in make_integrated_variance_model().identifier

    class OldModel:
        r"""Represent an explicitly retired q5 prescription so a variance factory must reject it.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        Prevent either variance factory from silently accepting the retired q5 prescription.
        Pass criteria: Both factories identify q10 defaults and reject a supplied q5-like model.
        Failure occurs when the stated comparison or expected rejection is not met. This
        controlled calculation does not calibrate a subhalo population against simulations or
        observations.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """

        wdm_power_convention = "published-q5"
        wdm_power_q = 5

    for factory in (make_variance_model, make_integrated_variance_model):
        with pytest.raises(ValueError, match="supported q10"):
            factory(OldModel())


def test_callable_adapter_uses_same_continuous_derivative() -> None:
    """Forward the same WDM variance slope used by the population model.

    Notes
    -----
    Pass criteria: The callable derivative equals the model derivative at two masses and
    z=2.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = ItamaeSubhalos(mass_wdm=_fixture_mass_wdm(), wdm_power_convention=STANDARD_T2_Q10)
    variance = make_variance_model(model)
    mass = np.array([100000000.0, 10000000000.0])
    np.testing.assert_array_equal(variance.dvariance_dmass(mass, 2.0), model.dsdm(mass, 2.0))


def test_integrated_sharp_k_path_is_sigma8_normalized_and_continuous() -> None:
    r"""Resolve the sharp-k mass assignment independently of the sigma8 window.

    Notes
    -----
    Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
    nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
    Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

    QUADPACK integrates linearly interpolated q10 table power over each
    original k interval, up to 2.5/R_th for three physical halo masses.
    Pass at rtol=2e-11 against the shared Gauss integral, with consistent
    density/k units and finite, distinct, negative slopes. The independent
    top-hat amplitude is checked in test_power_prescription.py. Thus the
    tolerance measures numerical integration agreement, not calibration.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.

    References
    ----------

    .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
        https://arxiv.org/abs/astro-ph/0501562
    """
    model = ItamaeSubhalos(mass_wdm=_fixture_mass_wdm(), wdm_power_convention=STANDARD_T2_Q10)
    variance = make_integrated_variance_model(model, n_k=1001)
    mass = np.geomspace(100000000.0, 100000000000000.0, 13)
    derivative = variance.dvariance_dmass(mass, 0.0)
    def integrand(logk):
        r"""Return dimensionless q10 table power per logarithmic wavenumber.

        Parameters
        ----------
        logk : float
            Natural logarithm of k in h/Mpc.

        Returns
        -------
        float
            Raw k**3 P_WDM(k)/(2*pi**2), before amplitude normalization.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        Used only for the independent adaptive cutoff integral in this
        test. The enclosing test states the numerical pass/failure limits.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        k = np.exp(logk)
        ratio = (1 + (model.a * k)**(2 * 1.12))**(-10 / 1.12)
        return k**3 * np.interp(k, k_file, Pk_file) * ratio / (2 * np.pi**2)

    for physical_mass in (1e8, 1e11, model.MassIn8Mpc / h):
        radius = (3 * physical_mass * h / (4 * np.pi * model.Rhomean_z))**(1 / 3)
        cutoff = min(2.5 / radius, k_file[-1])
        knots = np.log(np.r_[k_file[k_file < cutoff], cutoff])
        raw = sum(quad(integrand, left, right, epsabs=0, epsrel=2e-12)[0]
                  for left, right in zip(knots[:-1], knots[1:]))
        expected = raw / model._power_normalization()**2
        np.testing.assert_allclose(variance.variance(physical_mass), expected, rtol=2e-11)
    assert variance.rho_mean == pytest.approx(model.Rhomean_z * h**2)
    assert variance.k_min == pytest.approx(k_file[0] * h)
    assert variance.k_max == pytest.approx(k_file[-1] * h)
    assert np.all(np.isfinite(derivative))
    assert np.all(derivative < 0.0)
    assert np.unique(derivative).size == derivative.size
    assert "integrated-variance:v3" in variance.identifier
    assert "filter_scale=2.5" in variance.identifier
    assert "normalization=wdm-top-hat:R=8-Mpc/h:z=0:v1" in variance.identifier
    assert "wdm-power-ratio" in variance.identifier
    assert "convention=standard-t2-q10" in variance.identifier
    assert "formula-role=viel-transfer-amplitude-squared" in variance.identifier
    assert "q=10" in variance.identifier
    assert "source-k-unit=h/Mpc" in variance.identifier
    assert "unit-conversion=canonical-Mpc" in variance.identifier


def test_standard_q10_integrated_variance_formula_and_identifier_are_pinned() -> None:
    r"""Keep the current WDM prescription label and normalization identifiable.

    Notes
    -----
    Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
    nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
    Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

    Pass criteria: Sharp-k sigma(M8) differs from top-hat sigma8 and both
    adapters identify the v3 q10/half-mode prescription. The 5 percent
    separation rejects the old forced sharp-k normalization; it is not an
    uncertainty estimate. Independent amplitude accuracy is tested separately.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.

    References
    ----------

    .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
        https://arxiv.org/abs/astro-ph/0501562
    """
    model = ItamaeSubhalos(mass_wdm=_fixture_mass_wdm())
    integrated = make_integrated_variance_model(model, n_k=1001)
    assert abs(integrated.sigma(model.MassIn8Mpc / h, 0.0) / sigma_8 - 1) > 0.05
    for variance in [integrated, make_variance_model(model)]:
        assert "thermal-wdm-q10:2026-10-14:v3" in variance.identifier
        assert "convention=standard-t2-q10" in variance.identifier
        assert "formula-role=viel-transfer-amplitude-squared" in variance.identifier
        assert "q=10" in variance.identifier
        assert "half-mode=T_WDM/T_CDM=0.5;P_WDM/P_CDM=0.25" in variance.identifier


@pytest.mark.parametrize("convention", [STANDARD_T2_Q10])
def test_variance_and_derivative_use_one_continuous_integral(convention) -> None:
    """Use one continuous WDM density variance for both sigma and the EPS slope.

    Parameters
    ----------
    convention : str
        Declared WDM power-suppression convention.

    Notes
    -----
    Pass criteria: A centered mass finite difference agrees with dS/dM to rtol=4e-8 and
    callable/native sigma agrees; repeated queries retain the current cache behavior.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = ItamaeSubhalos(mass_wdm=2.0, wdm_power_convention=convention)
    mass = np.geomspace(1000000000.0, 10000000000000.0, 24)
    step = 0.0001
    upper, lower = (mass * np.exp(step), mass * np.exp(-step))
    numerical = (model.sigmaMz(upper, 0.5) ** 2 - model.sigmaMz(lower, 0.5) ** 2) / (upper - lower)
    np.testing.assert_allclose(model.dsdm(mass, 0.5), numerical, rtol=4e-08, atol=0.0)
    np.testing.assert_array_equal(model.sigmaMz(mass, 0.5), model.variance_model().sigma(mass, 0.5))
    variance = model.variance_model()
    model.sigmaMz(mass[::-1], 1.0)
    assert model.variance_model() is variance
    assert not hasattr(model, "_consistent_sigma_z0_values")


def test_lighter_wdm_has_more_small_scale_power_suppression() -> None:
    """Recover the expected particle-mass ordering of thermal WDM suppression.

    Notes
    -----
    Pass criteria: At 1e8 solar masses a 1-keV particle gives smaller sigma than 5 keV; at
    1e14 solar masses they agree within 0.3 percent.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    light = make_integrated_variance_model(
        mass_wdm=1.0, wdm_power_convention=STANDARD_T2_Q10, n_k=1001
    )
    heavy = make_integrated_variance_model(
        mass_wdm=5.0, wdm_power_convention=STANDARD_T2_Q10, n_k=1001
    )
    assert light.sigma(100000000.0, 0.0) < heavy.sigma(100000000.0, 0.0)
    assert light.sigma(100000000000000.0, 0.0) == pytest.approx(
        heavy.sigma(100000000000000.0, 0.0), rel=0.003
    )


def test_variance_cache_binds_wdm_particle_mass_and_round_trips(tmp_path) -> None:
    """Avoid reusing variance data for a different WDM mass or physical prescription.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.

    Notes
    -----
    Pass criteria: Changing particle mass or historical prescription changes the cache key;
    exact arrays reload and mismatched keys fail.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = ItamaeSubhalos(mass_wdm=_fixture_mass_wdm(), wdm_power_convention=STANDARD_T2_Q10)
    variance_model = make_integrated_variance_model(model, n_k=501)
    mass = np.geomspace(100000000.0, 1000000000000.0, 5)
    variance = variance_model.variance(mass)
    settings = {
        "calculation_specification": "sashimi-w:thermal-wdm-q10:2026-10-14:v3",
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
    # Alter the declared specification in a synthetic stale identifier;
    # this does not reconstruct or execute a historical power spectrum.
    old_identifier = variance_model.identifier.replace(
        "sashimi-w:thermal-wdm-q10:2026-10-14:v3:wdm-power-ratio:",
        "sashimi-w:thermal-wdm-q10:2026-09-11:v2:wdm-power-ratio:",
    )
    assert old_identifier != variance_model.identifier
    old_settings = {**settings, "calculation_specification": "sashimi-w:thermal-wdm-q10:2026-09-11:v2"}
    old_key = variance_cache_key(
        old_identifier,
        mass,
        backend_identifier=model.itamae_cosmology.identifier,
        settings=old_settings,
    )
    assert old_key != key
    cache = tmp_path / "variance.npz"
    save_variance_cache(cache, key=key, mass=mass, variance=variance)
    loaded_mass, loaded_variance = load_variance_cache(cache, expected_key=key)
    np.testing.assert_array_equal(loaded_mass, mass)
    np.testing.assert_array_equal(loaded_variance, variance)
    with pytest.raises(ValueError, match="key"):
        load_variance_cache(cache, expected_key=changed)
    with pytest.raises(ValueError, match="key"):
        load_variance_cache(cache, expected_key=old_key)


def test_integrated_power_is_a_snapshot_of_constructor_parameters():
    """Keep a saved variance object's spectrum fixed after the source model mutates.

    Notes
    -----
    Pass criteria: Changing the model suppression scale leaves the constructed power values
    and identifier exactly unchanged.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = ItamaeSubhalos(2.0)
    variance = make_integrated_variance_model(model, n_k=501)
    k = np.geomspace(0.01, 100, 15)
    before = variance.power(k).copy()
    identifier = variance.identifier
    model.a *= 2
    np.testing.assert_array_equal(variance.power(k), before)
    assert variance.identifier == identifier
