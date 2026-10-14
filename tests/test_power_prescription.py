"""Independent checks of the physical WDM spectrum and halo mass assignment."""

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import spherical_jn

import sashimi_w as w
from sashimi_w._power import sigma8_normalization, top_hat_window


def test_top_hat_zero_and_bessel_reference():
    """Keep the physical top-hat window finite at zero and at small k R.

    Notes
    -----
    Pass when W(0)=1 exactly and nonzero values agree with the independent
    spherical-Bessel identity 3*j1(x)/x to absolute error 2e-12. Failure
    detects cancellation or an incorrect smoothing window. This tolerance
    concerns floating-point evaluation, not a physical calibration.
    """
    x = np.array([0.0, 1e-10, 1e-4, 0.01, 0.1, 1.0, 8.0, 100.0])
    expected = np.ones_like(x)
    expected[1:] = 3 * spherical_jn(1, x[1:]) / x[1:]
    assert top_hat_window(0.0) == 1.0
    np.testing.assert_allclose(top_hat_window(x), expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("particle_mass", [0.5, 2.0, 5.0])
def test_eps_power_has_independent_top_hat_sigma8(particle_mass):
    """Measure sigma8 from the actual EPS spectrum using an independent rule.

    Parameters
    ----------
    particle_mass : float
        Thermal WDM mass in keV, spanning stronger and weaker suppression.

    Notes
    -----
    Integrate normalized EPS power with Simpson quadrature and the Bessel
    top-hat at R=8 Mpc/h on 50,001 and 100,001 log-k nodes. The domain
    1e-7..1e4 h/Mpc excludes only negligible tails for these three masses.
    Pass when both resolutions reproduce the configured sigma8 to relative
    error 2e-8 and agree with each other. A separate sharp-k normalization,
    wrong h conversion or missing amplitude factor fails. This verifies
    the declared spectrum amplitude, not subhalo population calibration.
    """
    model = w.subhalos(particle_mass)
    results = []
    for size in (50001, 100001):
        logk = np.linspace(np.log(1e-7), np.log(1e4), size)
        k = np.exp(logk)
        power = model.variance_model().power(k * w.h) * w.h**3
        window = 3 * spherical_jn(1, 8 * k) / (8 * k)
        results.append(np.sqrt(simpson(k**3 * power * window**2 / (2 * np.pi**2), x=logk)))
    np.testing.assert_allclose(results, w.sigma_8, rtol=2e-8, atol=0)
    np.testing.assert_allclose(results[0], results[1], rtol=2e-8, atol=0)
    finer = sigma8_normalization(w.k_file, w.Pk_file, model.a, w.sigma_8, order=32)
    np.testing.assert_allclose(model._power_normalization(), finer, rtol=2e-12, atol=0)
    # A different smoothing window has a different rms at the same physical mass.
    assert abs(model.sigmaMz(model.MassIn8Mpc / w.h, 0.0) / w.sigma_8 - 1) > 0.05


@pytest.mark.parametrize("bad", ["negative-power", "zero-power", "unordered-k", "zero-target"])
def test_invalid_normalization_cannot_create_a_physical_spectrum(bad):
    """Reject inputs that cannot define a positive normalized density field.

    Parameters
    ----------
    bad : str
        One invalid table or target condition in an otherwise valid setup.

    Notes
    -----
    Pass only if the invalid case raises ValueError and the original
    positive table returns a finite divisor. This guards against negative
    or undefined variances being silently propagated to EPS abundances;
    it does not impose additional constraints on valid cosmologies.
    """
    k, power, target = np.array([0.01, 0.1, 1.0]), np.ones(3), 0.82
    assert np.isfinite(sigma8_normalization(k, power, 0.03, target))
    if bad == "negative-power":
        power[1] = -1
    elif bad == "zero-power":
        power[:] = 0
    elif bad == "unordered-k":
        k = k[::-1]
    else:
        target = 0
    with pytest.raises(ValueError):
        sigma8_normalization(k, power, 0.03, target)
