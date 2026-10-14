"""Independent checks for the numerical EPS backport, with no fixture rewriting."""

import numpy as np
import pytest
from scipy.integrate import quad

from sashimi_w import Subhalos
from sashimi_w._physics import _normalized_eps_kernel


@pytest.fixture(scope="module")
def model():
    """Construct the 2-keV q10 WDM model for controlled EPS support and variance-gap checks.

    Returns
    -------
    Subhalos
        Fixed particle-mass model using the standard canonical variance prescription.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    return Subhalos(2.0)


@pytest.mark.parametrize("gap", [0.0, 1e-14, 1e-6, 9.999e-5, 1.001e-4, 0.1, 2.0, 10.0])
def test_kernel_matches_independent_normalization_integral(gap):
    """Recover the normalized WDM EPS kernel through the zero-barrier-gap limit.

    Parameters
    ----------
    gap : float
        Collapse-barrier difference in the controlled EPS normalization case.

    Notes
    -----
    Pass criteria: For gaps spanning zero and the small-gap branch boundary, the kernel
    agrees with independent definite quadrature to rtol=3e-14.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    dsmin, ds = 0.5, np.array([0.5, 0.7, 3.0, 20.0])
    ratio = gap / np.sqrt(2 * dsmin)
    # Substitute u=delta/sqrt(2*S) in the fixed-gap EPS normalizer,
    # then scale to [0, 1]. The common delta cancels before evaluation.
    normalizer = quad(lambda u: np.exp(-((ratio * u) ** 2)), 0, 1, epsabs=1e-13)[0]
    expected = np.sqrt(dsmin) / (2 * ds**1.5) * np.exp(-(gap**2) / (2 * ds)) / normalizer
    np.testing.assert_allclose(_normalized_eps_kernel(gap, ds, dsmin), expected, rtol=3e-14)


@pytest.mark.parametrize("gap", [0.0, 1e-12, 0.1, 3.0])
def test_fixed_gap_kernel_integrates_to_one(gap):
    """Normalize conditional WDM accretion probability over its full variance support.

    Parameters
    ----------
    gap : float
        Collapse-barrier difference in the controlled EPS normalization case.

    Notes
    -----
    Pass criteria: After mapping the infinite variance interval to [0,1], the integrated
    kernel equals one to rtol=2e-12.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    dsmin = 0.7
    # ds=dsmin/u**2 maps [dsmin,infinity) onto u in [1,0].
    integral = quad(
        lambda u: float(_normalized_eps_kernel(gap, dsmin / u**2, dsmin)) * 2 * dsmin / u**3,
        0,
        1,
        epsabs=2e-12,
    )[0]
    assert integral == pytest.approx(1.0, rel=2e-12)


def test_zero_limit_on_tiny_variance_gap_and_unrepresentable_exponent():
    """Avoid undefined arithmetic in extreme normalized EPS limits.

    Notes
    -----
    Pass criteria: A zero barrier gap at variance 1e-250 gives 5e249 to rtol=1e-13, while an
    overwhelming exponential suppression returns zero without invalid/divide/overflow
    errors.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        assert _normalized_eps_kernel(0.0, 1e-250, 1e-250) == pytest.approx(5e249, rel=1e-13)
        assert _normalized_eps_kernel(1e100, 1e-120, 1e-120) == 0.0


@pytest.mark.parametrize("gap,ds,dsmin", [(-1, 2, 1), (1, 0, 1), (1, 2, 0), (np.nan, 2, 1)])
def test_active_domain_errors_are_explicit(gap, ds, dsmin):
    """Reject physically invalid active EPS barrier or variance gaps.

    Parameters
    ----------
    gap : float
        Collapse-barrier difference in the controlled EPS normalization case.
    ds : object
        Controlled input supplied by the parametrization or fixture for the specific
        physical/reference case described below.
    dsmin : object
        Controlled input supplied by the parametrization or fixture for the specific
        physical/reference case described below.

    Notes
    -----
    Pass criteria: Negative barrier gap, nonpositive variance support or NaN raises
    ValueError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    with pytest.raises(ValueError, match="WDM EPS"):
        _normalized_eps_kernel(gap, ds, dsmin)


def independent_gap(variance, small, large):
    """Adaptive quadrature of the physical interval, split only at spectrum knots.

    Parameters
    ----------
    variance : IntegratedVarianceModel
        Canonical spectrum/density/window configuration being checked against an independent
        integration method.
    small : float or array_like
        Positive ordered halo masses in physical Msun; smaller_mass <= larger_mass defines a
        nonnegative sharp-k variance interval.
    large : float or array_like
        Positive ordered halo masses in physical Msun; smaller_mass <= larger_mass defines a
        nonnegative sharp-k variance interval.

    Returns
    -------
    float
        Adaptive-quadrature S(small)-S(large), split at actual spectrum knots rather than
        reusing the production fixed cells.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """

    def cutoff(mass):
        """Compute the physical sharp-k cutoff independently for the reference integral.

        Parameters
        ----------
        mass : float or array_like
            Positive halo mass. Use solar masses with the canonical backend; a supplied variance
            prescription must use the same mass convention and factors of h.

        Returns
        -------
        float
            Cutoff in Mpc^-1 clipped to the finite spectrum domain.

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        radius = (3 * mass / (4 * np.pi * variance.rho_mean)) ** (1 / 3)
        return np.clip(variance.filter_scale / radius, variance.k_min, variance.k_max)

    left, right = cutoff(large), cutoff(small)
    knots = variance.power.integration_breakpoints
    edges = np.log(np.r_[left, knots[(knots > left) & (knots < right)], right])

    def integrand(logk):
        """Evaluate k^3 P(k)/(2*pi^2) for independent log-wavenumber quadrature.

        Parameters
        ----------
        logk : float
            Natural logarithm of canonical wavenumber k in Mpc^-1.

        Returns
        -------
        float
            Dimensionless variance contribution per unit ln(k).

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        k = np.exp(logk)
        return k**3 * float(variance.power(k)) / (2 * np.pi**2)

    return sum(
        quad(integrand, a, b, epsabs=0.0, epsrel=2e-12)[0]
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    )


@pytest.mark.parametrize("small,large", [(1e12, 2e12), (1e8, 2e8), (1e2, 2e2), (1e8, 1.000001e8)])
def test_variance_gap_matches_independent_interval_integral(model, small, large):
    """Resolve the variance difference that controls WDM EPS accretion.

    Parameters
    ----------
    model : Subhalos
        Shared 2-keV WDM model used for the controlled EPS and variance queries.
    small : float
        Smaller halo mass in solar masses defining the upper wavenumber of the variance
        interval.
    large : float
        Larger halo mass in solar masses defining the lower wavenumber of the variance
        interval.

    Notes
    -----
    Pass criteria: Four widely separated or nearly equal mass pairs give positive gaps
    matching independent interval quadrature to rtol=3e-12.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    expected = independent_gap(model.variance_model(), small, large)
    actual = model._variance_gap(small, large)
    assert actual > 0
    assert actual == pytest.approx(expected, rel=3e-12)


def test_saturated_tail_preserves_positive_gap_and_ordering(model):
    """Keep low-mass EPS support even when full variances round to the same float.

    Parameters
    ----------
    model : Subhalos
        Shared 2-keV WDM model used for the controlled EPS and variance queries.

    Notes
    -----
    Pass criteria: The 100/200-solar-mass interval remains positive despite equal rounded
    full variances; equal masses give zero, reversed ordering fails and broadcasting agrees
    with scalar calls.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    assert model.sigmaMz(100.0, 0.0) ** 2 == model.sigmaMz(200.0, 0.0) ** 2
    assert model._variance_gap(100.0, 200.0) > 0.0
    assert model._variance_gap(100.0, 100.0) == 0.0
    np.testing.assert_array_equal(model._variance_gap([], []), [])
    with pytest.raises(ValueError, match="ordered"):
        model._variance_gap(200.0, 100.0)
    small = np.array([[1e2], [1e8]])
    large = np.array([1e10, 1e12])
    result = model._variance_gap(small, large)
    assert result.shape == (2, 2)
    for i, j in np.ndindex(result.shape):
        assert result[i, j] == model._variance_gap(small[i, 0], large[j])


@pytest.mark.parametrize("order", [1, 4, 64, 200])
def test_eps_shapes_support_and_finite_low_mass_tails(model, order):
    """Preserve EPS support across host quadrature resolution and mass-grid representations.

    Parameters
    ----------
    model : Subhalos
        Shared 2-keV WDM model used for the controlled EPS and variance queries.
    order : int
        Quadrature or recursive-hierarchy order under test; malformed cases exercise
        validation.

    Notes
    -----
    Pass criteria: Orders 1,4,64,200 give finite nonnegative, correctly shaped rates; matrix
    and vector mass queries agree and forbidden high-mass nodes have zero weight.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    z = np.array([0.25, 0.5, 1.0])
    mass = np.logspace(1, 11, 5)
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        flat = model.Na_calc(mass, z, 1e12, N_herm=order)
        grid = model.Na_calc(np.broadcast_to(mass, (3, 5)), z, 1e12, N_herm=order)
        forbidden = model.Na_calc(np.array([5e11, 1e12]), z, 1e12, N_herm=order)
    assert grid.shape == (3, 5)
    assert np.all(np.isfinite(grid)) and np.all(grid >= 0)
    np.testing.assert_array_equal(grid, flat)
    np.testing.assert_array_equal(forbidden, np.zeros((3, 2)))


@pytest.mark.parametrize("sigmafac", [-0.5, 0.0, 0.5])
def test_deterministic_rows_do_not_depend_on_redshift_grid(model, sigmafac):
    """Keep a local accretion rate independent of unrelated later query epochs.

    Parameters
    ----------
    model : Subhalos
        Shared 2-keV WDM model used for the controlled EPS and variance queries.
    sigmafac : float
        Offset in units of the lognormal host-history scatter.

    Notes
    -----
    Pass criteria: Appending extra redshifts leaves the original rates exactly unchanged for
    three host-scatter offsets.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    mass = np.logspace(7, 10, 4)
    z = np.array([0.25, 0.5, 1.0])
    short = model.Na_calc(mass, z, 1e12, N_herm=1, sigmafac=sigmafac)
    for extra in [1.01, 1.5, 3.0]:
        longer = model.Na_calc(mass, np.r_[z, extra], 1e12, N_herm=1, sigmafac=sigmafac)
        np.testing.assert_array_equal(short, longer[:3])


def test_exact_z1_deterministic_scatter_anchor(monkeypatch):
    """Use the specified z=1 host-history scatter anchor exactly.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: The variance-gap upper masses match the independently evaluated z=1
    anchored host formula to rtol=2e-15.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = Subhalos(2.0)
    z, sigmafac, host = np.array([0.25, 0.5, 1.0, 1.5]), -0.5, 1e12
    observed = []
    original = model._variance_gap

    def spy(small, large):
        """Record the host-mass arguments reaching variance-gap evaluation without changing the
        calculation.

        Parameters
        ----------
        small : float or array_like
            Positive ordered halo masses in physical Msun; smaller_mass <= larger_mass defines a
            nonnegative sharp-k variance interval.
        large : float or array_like
            Positive ordered halo masses in physical Msun; smaller_mass <= larger_mass defines a
            nonnegative sharp-k variance interval.

        Returns
        -------
        numpy.ndarray
            Unchanged production variance interval, with the observed host mass saved for the
            exact-z=1 check.

        Notes
        -----
        Use the specified z=1 host-history scatter anchor exactly. Pass criteria: The
        variance-gap upper masses match the independently evaluated z=1 anchored host formula to
        rtol=2e-15. Failure occurs when the stated comparison or expected rejection is not met.
        This controlled calculation does not calibrate a subhalo population against simulations
        or observations.
        """
        observed.append(np.asarray(large))
        return original(small, large)

    monkeypatch.setattr(model, "_variance_gap", spy)
    model.Na_calc(np.array([1e8]), z, host, N_herm=1, sigmafac=sigmafac)
    mean = model.Mzzi(host, z, 0.0)
    at_one = model.Mzzi(host, 1.0, 0.0)
    width_at_one = 0.12 - 0.15 * np.log10(at_one / host)
    width = np.where(
        z <= 1.0,
        width_at_one * np.log10(mean / host) / np.log10(at_one / host),
        0.12 - 0.15 * np.log10(mean / host),
    )
    scattered = 10 ** (np.log10(mean) + sigmafac * width)
    expected_max = np.minimum(mean + np.minimum(scattered, host / 2), host)
    np.testing.assert_allclose(observed[0], expected_max, rtol=2e-15)


def test_active_zero_gap_is_not_silently_erased(monkeypatch, model):
    """Retain a finite accretion contribution when the collapse barrier gap is zero.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.
    model : Subhalos
        Shared 2-keV WDM model used for the controlled EPS and variance queries.

    Notes
    -----
    Pass criteria: For an exactly zero barrier prescription, active rates remain finite and
    strictly positive.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    monkeypatch.setattr(
        model,
        "delc_Y11",
        lambda mass, z: np.zeros(np.broadcast_shapes(np.shape(mass), np.shape(z))),
    )
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        rate = model.Na_calc(np.array([1e8, 1e9]), np.array([0.5]), 1e12, N_herm=1)
    assert np.all(np.isfinite(rate)) and np.all(rate > 0.0)


def test_catalog_records_numerical_backport(model):
    """Identify the EPS numerical prescription used in a saved WDM result.

    Parameters
    ----------
    model : Subhalos
        Shared 2-keV WDM model used for the controlled EPS and variance queries.

    Notes
    -----
    Pass criteria: Metadata record the exact-z=1 anchor and direct-variance-gap
    normalized-zero-limit implementation.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    metadata = model._catalog_metadata().as_mapping()
    assert metadata["deterministic_scatter_anchor"] == "exact-z=1"
    assert metadata["eps_numerics"] == "active-support:direct-variance-gap:normalized-zero-limit:v1"


def test_wholly_suppressed_catalog_preserves_zero_population():
    """Represent a physically empty WDM sample without NaN normalization.

    Notes
    -----
    Pass criteria: The strongly suppressed 0.5-keV case has zero base/final weights and
    finite physical columns under strict floating-point handling.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        catalog = Subhalos(0.5).rs_rhos_catalog_calc(
            M0=1e12,
            zmax=1.0,
            dz=0.5,
            N_ma=4,
            logmamin=6.0,
            logmamax=8.0,
            N_herm=2,
            N_hermNa=3,
        )
    assert np.all(catalog.weight_final == 0.0)
    assert np.all(catalog.weights["weight_base"] == 0.0)
    assert all(np.all(np.isfinite(value)) for value in catalog.columns.values())


@pytest.mark.parametrize("bad_rate", [-1.0, np.nan, np.inf])
def test_invalid_population_normalization_raises(monkeypatch, bad_rate):
    """Reject a catalogue whose EPS rates cannot define a nonnegative finite measure.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.
    bad_rate : object
        Controlled input supplied by the parametrization or fixture for the specific
        physical/reference case described below.

    Notes
    -----
    Pass criteria: Injected negative, NaN or infinite accretion rates raise a normalization
    error.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = Subhalos(2.0)
    monkeypatch.setattr(
        model, "Na_calc", lambda mass, *args, **kwargs: np.full_like(mass, bad_rate)
    )
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="normalization"):
        model.rs_rhos_catalog_calc(
            M0=1e12, zmax=1.0, dz=0.5, N_ma=4, logmamin=6.0, logmamax=8.0, N_herm=2, N_hermNa=3
        )
