"""Independent checks for the numerical EPS backport, with no fixture rewriting."""

import numpy as np
import pytest
from scipy.integrate import quad

from sashimi_w import Subhalos
from sashimi_w._physics import _normalized_eps_kernel


@pytest.fixture(scope="module")
def model():
    return Subhalos(2.0)


@pytest.mark.parametrize("gap", [0.0, 1e-14, 1e-6, 9.999e-5, 1.001e-4, 0.1, 2.0, 10.0])
def test_kernel_matches_independent_normalization_integral(gap):
    dsmin, ds = 0.5, np.array([0.5, 0.7, 3.0, 20.0])
    ratio = gap / np.sqrt(2 * dsmin)
    # Substitute u=delta/sqrt(2*S) in the fixed-gap EPS normalizer,
    # then scale to [0, 1]. The common delta cancels before evaluation.
    normalizer = quad(lambda u: np.exp(-(ratio * u) ** 2), 0, 1, epsabs=1e-13)[0]
    expected = np.sqrt(dsmin) / (2 * ds**1.5) * np.exp(-gap**2 / (2 * ds)) / normalizer
    np.testing.assert_allclose(_normalized_eps_kernel(gap, ds, dsmin), expected, rtol=3e-14)


@pytest.mark.parametrize("gap", [0.0, 1e-12, 0.1, 3.0])
def test_fixed_gap_kernel_integrates_to_one(gap):
    dsmin = 0.7
    # ds=dsmin/u**2 maps [dsmin,infinity) onto u in [1,0].
    integral = quad(
        lambda u: float(_normalized_eps_kernel(gap, dsmin / u**2, dsmin)) * 2 * dsmin / u**3,
        0, 1, epsabs=2e-12,
    )[0]
    assert integral == pytest.approx(1.0, rel=2e-12)


def test_zero_limit_on_tiny_variance_gap_and_unrepresentable_exponent():
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        assert _normalized_eps_kernel(0.0, 1e-250, 1e-250) == pytest.approx(5e249, rel=1e-13)
        assert _normalized_eps_kernel(1e100, 1e-120, 1e-120) == 0.0


@pytest.mark.parametrize("gap,ds,dsmin", [(-1, 2, 1), (1, 0, 1), (1, 2, 0), (np.nan, 2, 1)])
def test_active_domain_errors_are_explicit(gap, ds, dsmin):
    with pytest.raises(ValueError, match="WDM EPS"):
        _normalized_eps_kernel(gap, ds, dsmin)


def independent_gap(variance, small, large):
    """Adaptive quadrature of the physical interval, split only at spectrum knots."""
    def cutoff(mass):
        radius = (3 * mass / (4 * np.pi * variance.rho_mean)) ** (1 / 3)
        return np.clip(variance.filter_scale / radius, variance.k_min, variance.k_max)
    left, right = cutoff(large), cutoff(small)
    knots = variance.power.integration_breakpoints
    edges = np.log(np.r_[left, knots[(knots > left) & (knots < right)], right])
    def integrand(logk):
        k = np.exp(logk)
        return k**3 * float(variance.power(k)) / (2 * np.pi**2)
    return sum(quad(integrand, a, b, epsabs=0.0, epsrel=2e-12)[0]
               for a, b in zip(edges[:-1], edges[1:], strict=True))


@pytest.mark.parametrize("small,large", [(1e12, 2e12), (1e8, 2e8), (1e2, 2e2), (1e8, 1.000001e8)])
def test_variance_gap_matches_independent_interval_integral(model, small, large):
    expected = independent_gap(model.variance_model(), small, large)
    actual = model._variance_gap(small, large)
    assert actual > 0
    assert actual == pytest.approx(expected, rel=3e-12)


def test_saturated_tail_preserves_positive_gap_and_ordering(model):
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
    mass = np.logspace(7, 10, 4)
    z = np.array([0.25, 0.5, 1.0])
    short = model.Na_calc(mass, z, 1e12, N_herm=1, sigmafac=sigmafac)
    for extra in [1.01, 1.5, 3.0]:
        longer = model.Na_calc(mass, np.r_[z, extra], 1e12, N_herm=1, sigmafac=sigmafac)
        np.testing.assert_array_equal(short, longer[:3])


def test_exact_z1_deterministic_scatter_anchor(monkeypatch):
    model = Subhalos(2.0)
    z, sigmafac, host = np.array([0.25, 0.5, 1.0, 1.5]), -0.5, 1e12
    observed = []
    original = model._variance_gap
    def spy(small, large):
        observed.append(np.asarray(large))
        return original(small, large)
    monkeypatch.setattr(model, "_variance_gap", spy)
    model.Na_calc(np.array([1e8]), z, host, N_herm=1, sigmafac=sigmafac)
    mean = model.Mzzi(host, z, 0.0)
    at_one = model.Mzzi(host, 1.0, 0.0)
    width_at_one = 0.12 - 0.15 * np.log10(at_one / host)
    width = np.where(z <= 1.0, width_at_one * np.log10(mean / host) / np.log10(at_one / host),
                     0.12 - 0.15 * np.log10(mean / host))
    scattered = 10 ** (np.log10(mean) + sigmafac * width)
    expected_max = np.minimum(mean + np.minimum(scattered, host / 2), host)
    np.testing.assert_allclose(observed[0], expected_max, rtol=2e-15)


def test_active_zero_gap_is_not_silently_erased(monkeypatch, model):
    monkeypatch.setattr(model, "delc_Y11", lambda mass, z: np.zeros(np.broadcast_shapes(np.shape(mass), np.shape(z))))
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        rate = model.Na_calc(np.array([1e8, 1e9]), np.array([0.5]), 1e12, N_herm=1)
    assert np.all(np.isfinite(rate)) and np.all(rate > 0.0)


def test_catalog_records_numerical_backport(model):
    metadata = model._catalog_metadata().as_mapping()
    assert metadata["deterministic_scatter_anchor"] == "exact-z=1"
    assert metadata["eps_numerics"] == "active-support:direct-variance-gap:normalized-zero-limit:v1"


def test_wholly_suppressed_catalog_preserves_zero_population():
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        catalog = Subhalos(0.5).rs_rhos_catalog_calc(
            M0=1e12, zmax=1.0, dz=0.5, N_ma=4, logmamin=6.0, logmamax=8.0,
            N_herm=2, N_hermNa=3,
        )
    assert np.all(catalog.weight_final == 0.0)
    assert np.all(catalog.weights["weight_base"] == 0.0)
    assert all(np.all(np.isfinite(value)) for value in catalog.columns.values())


@pytest.mark.parametrize("bad_rate", [-1.0, np.nan, np.inf])
def test_invalid_population_normalization_raises(monkeypatch, bad_rate):
    model = Subhalos(2.0)
    monkeypatch.setattr(model, "Na_calc", lambda mass, *args, **kwargs: np.full_like(mass, bad_rate))
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="normalization"):
        model.rs_rhos_catalog_calc(M0=1e12, zmax=1.0, dz=0.5, N_ma=4,
                                   logmamin=6.0, logmamax=8.0, N_herm=2, N_hermNa=3)
