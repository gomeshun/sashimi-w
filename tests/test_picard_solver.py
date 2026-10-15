"""Boundary, state identity, and analytic tests of the standalone solver."""
from pathlib import Path
import importlib
import warnings
import numpy as np
import pytest
from scipy.optimize import brentq

from picard_tidal_stripping import (PicardTidalStrippingTable, endpoint_mass,
    history_mass, direct_log_mass, PicardFallbackWarning)


class AnalyticHost:
    def __init__(self, exponent=.01):
        self.M0=1e12
        self.z_min=0.
        self.z_max=7.
        self.exponent=exponent
        self.rate=.5
    def Mzvir(self,z):
        return np.ones_like(np.asarray(z))*self.M0
    def zetaMz(self,z):
        return np.ones_like(np.asarray(z))*self.exponent
    def Phi(self,z):
        return np.ones_like(np.asarray(z))*self.rate
    def exact(self,ma,za,z):
        if self.exponent==0:
            return ma*np.exp(self.rate*(z-za))
        return ma*(1-self.exponent*self.rate*(ma/self.M0)**self.exponent*(z-za))**(-1/self.exponent)


@pytest.mark.parametrize('exponent',[-.005,0.,.02])
def test_full_history_against_analytic_unexpanded_equation(exponent):
    solver=AnalyticHost(exponent)
    mass=np.array([1e-9,1e3,1e9])
    times=np.linspace(7.,0.,100)
    expected=solver.exact(mass[None,:],7.,times[:,None])
    actual=history_mass(solver,mass,7.,times,n_iterations=4,n_integration=257)
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=0)
    np.testing.assert_array_equal(actual[0],mass)
    assert not solver._picard_events


def test_shapes_no_evolution_and_partial_history_start():
    solver=AnalyticHost()
    assert isinstance(endpoint_mass(solver,1e6,0.,0.),float)
    np.testing.assert_array_equal(endpoint_mass(solver,[1.,2.],0.,0.),[1.,2.])
    assert endpoint_mass(solver,np.empty((0,2)),0.,0.).shape==(0,2)
    assert history_mass(solver,[],1.,[1.,.5,0.]).shape==(3,0)
    assert history_mass(solver,[1.,2.],1.,[]).shape==(0,2)
    assert history_mass(solver,np.ones((2,3)),1.,[1.,1.]).shape==(2,2,3)
    times=np.array([2.,1.,0.])
    actual=history_mass(solver,[1e6],3.,times,n_iterations=4,n_integration=257)
    np.testing.assert_allclose(actual[:,0],solver.exact(1e6,3.,times),rtol=1e-5)
    np.testing.assert_allclose(direct_log_mass(solver,[1e6],3.,times)[:,0],solver.exact(1e6,3.,times),rtol=1e-10)


@pytest.mark.parametrize('ma,za,z',[(0.,1.,0.),(-1.,1.,0.),(np.nan,1.,0.),
    (np.inf,1.,0.),(1.,np.nan,0.),(1.,1.,np.inf),(1.,0.,1.),(1.,0.,-1.)])
def test_rejects_nonphysical_solver_coordinates(ma,za,z):
    for method in [endpoint_mass,history_mass,direct_log_mass]:
        with pytest.raises(ValueError):
            method(AnalyticHost(),ma,za,z)


def test_history_direction_and_unknown_options_are_not_silently_ignored():
    solver=AnalyticHost()
    for method in [history_mass,direct_log_mass]:
        with pytest.raises(ValueError):
            method(solver,[1e6],3.,[3.,1.,2.])
    for method in [endpoint_mass,history_mass]:
        with pytest.raises(TypeError):
            method(solver,1e6,3.,0.,unknown_parameter=1)
        with pytest.raises(ValueError):
            method(solver,1e6,3.,0.,n_iterations=0)
        with pytest.raises(ValueError):
            method(solver,1e6,3.,0.,n_integration=1.5)


def test_table_cache_options_and_host_background_invalidation():
    solver=AnalyticHost()
    endpoint_mass(solver,[1e4,1e6],3.,0.)
    table=next(iter(solver._picard_tables.values()))
    endpoint_mass(solver,1e5,3.,0.)
    endpoint_mass(solver,1e5,3.,0.,n_iterations=4)
    solver.M0*=10
    with pytest.raises(ValueError,match='stale'):
        table.mass(1e5,3.)
    second=endpoint_mass(solver,[1e4,1e6],3.,0.)
    np.testing.assert_allclose(second, solver.exact(np.array([1e4,1e6]),3.,0.), rtol=1e-4)
    solver.rate=.7
    third=endpoint_mass(solver,[1e4,1e6],3.,0.)
    np.testing.assert_allclose(third, solver.exact(np.array([1e4,1e6]),3.,0.), rtol=1e-4)


def test_no_silent_extrapolation_and_counted_fallback():
    solver=AnalyticHost()
    table=PicardTidalStrippingTable(solver,log10_ratio_min=-12.,log10_ratio_max=-1.)
    with pytest.raises(ValueError):
        table.mass(1e-3,3.)
    with pytest.warns(PicardFallbackWarning):
        actual=endpoint_mass(solver,1e-15,3.,0.)
    np.testing.assert_allclose(actual,solver.exact(1e-15,3.,0.),rtol=1e-10)
    assert solver._picard_events[-1]['mass_count']==1
    with pytest.warns(PicardFallbackWarning):
        endpoint_mass(solver,1e6,7.,0.,n_iterations=1,convergence_tolerance=1e-15)
    assert 'convergence' in solver._picard_events[-1]['reason']


def _wdm_host_breakpoints(solver, zmax):
    """Locate narrow features of the retained WDM concentration lookup.

    Parameters
    ----------
    solver : sashimi_w.TidalStrippingSolver
        WDM host and stripping provider; the controlled test uses 2 keV,
        a 1e12 Msun host and the deterministic mean history.
    zmax : float
        Positive upper redshift of integration toward z=0.

    Returns
    -------
    list of float
        Descending redshifts at nearest-mass changes and real-formation
        trial boundaries, plus a point 1e-6 above each formation boundary.

    Notes
    -----
    The first valid future-formation trial can produce a narrow feature.
    An interior point forces the independent ODE to sample it; it does not
    alter the concentration rule. The enclosing test checks refinement and
    comparison tolerances. These points are specific to the retained grids,
    not a calibration or a general accuracy guarantee for arbitrary hosts.
    """
    import sashimi_w as w
    model = solver.model
    low = model.Mzzi(solver.M0, zmax, 0.0) * w.h
    high = solver.M0 * w.h

    def offset(redshift, boundary):
        """Measure the host's table mass relative to a lookup transition.

        Parameters
        ----------
        redshift : float
            Host redshift in [0, zmax].
        boundary : float
            Midpoint between adjacent mass nodes in numerical Msun/h.

        Returns
        -------
        float
            Host table mass minus boundary in numerical Msun/h.

        Notes
        -----
        A sign change brackets a nearest-mass switch, not a physical event.
        """
        return model.Mzzi(solver.M0, redshift, 0.0) * w.h - boundary

    mass_boundaries = (model.filter_Mass[1:] + model.filter_Mass[:-1]) / 2
    points = [brentq(offset, 0.0, zmax, args=(boundary,), xtol=1e-14)
              for boundary in mass_boundaries if low < boundary < high]
    c = 10**(np.arange(100) * 4 / 99)
    fraction = (np.log(2) - 0.5) / (np.log(1 + c) - c / (1 + c))
    density_ratio = c**3 * fraction / (650 / 200)
    trial_z = np.cbrt((w.OmegaL / density_ratio - w.OmegaL) / w.OmegaM) - 1
    for redshift in trial_z[(trial_z > 0) & (trial_z < zmax - 1e-6)]:
        points.extend((redshift, redshift + 1e-6))
    return sorted(set(points), reverse=True)


def _segmented_log_reference(solver, mass, points, tolerance):
    """Integrate the unchanged tidal rate while resolving known lookup features.

    Parameters
    ----------
    solver : sashimi_w.TidalStrippingSolver
        Physical WDM stripping-rate provider.
    mass : array_like
        Initial physical masses in Msun.
    points : sequence of float
        Strictly descending redshifts including initial and final times.
    tolerance : float
        Local relative and absolute DOP853 tolerances in log mass.

    Returns
    -------
    numpy.ndarray
        Final physical masses in Msun with the input shape.

    Notes
    -----
    Restart only the numerical integration at the specified boundaries.
    The enclosing test checks a tenfold tolerance refinement; a local ODE
    tolerance alone is not a bound on integration across narrow features.
    """
    result = np.asarray(mass).copy()
    for upper, lower in zip(points[:-1], points[1:]):
        result = np.asarray(direct_log_mass(solver, result, upper, lower,
                                           rtol=tolerance, atol=tolerance))
    return result


def test_public_dispatcher_candidate_and_legacy_paths():
    """Compare Picard and legacy mass evolution with resolved direct integration.

    Notes
    -----
    Evolve 1e5/1e8 Msun subhalos from z=3 to 0 in a 1e12 Msun host.
    For WDM, resolve the retained concentration lookup's narrow features:
    segmented log-mass DOP853 at 1e-11/1e-12 must agree within 2e-8, and
    physical-mass LSODA uses a maximum redshift step of 2e-4. Pass at the
    original relative limits 1e-3 for Picard and 2e-7 for LSODA. Unsupported
    options must still fail. This checks these numerical controls on one
    host, not default-ODE accuracy or physical calibration.
    """
    variant=next(v for v in ('c','si','w','f') if (Path(__file__).resolve().parents[1]/('sashimi_'+v+'.py')).exists())
    module=importlib.import_module('sashimi_'+variant)
    if variant=='w':
        solver=module.TidalStrippingSolver(module.subhalos(2.),1e12,z_max=3.)
    elif variant=='f':
        solver=module.TidalStrippingSolver(1e12,1.,z_max=3.)
    else:
        solver=module.TidalStrippingSolver(1e12,z_max=3.)
    method='picard' if variant=='si' else 'picard_table'
    mass=np.array([1e5,1e8])
    output=solver.subhalo_mass_stripped(mass,3.,0.,method=method)
    if variant == 'w':
        points = [3.0] + _wdm_host_breakpoints(solver, 3.0) + [0.0]
        expected = _segmented_log_reference(solver, mass, points, 1e-11)
        refined = _segmented_log_reference(solver, mass, points, 1e-12)
        np.testing.assert_allclose(expected, refined, rtol=2e-8)
        legacy_options = dict(hmax=2e-4, mxstep=10000)
    else:
        expected=direct_log_mass(solver,mass,3.,0.)
        legacy_options = {}
    np.testing.assert_allclose(np.asarray(output).ravel(),expected,rtol=1e-3)
    legacy=solver.subhalo_mass_stripped(mass,3.,0.,method='odeint',rtol=1e-10,atol=1e-10,**legacy_options)
    np.testing.assert_allclose(np.asarray(legacy).ravel(),expected,rtol=2e-7)
    with pytest.raises(TypeError):
        solver.subhalo_mass_stripped(mass,3.,0.,method=method,mxstep=10)
    if variant!='w':
        with pytest.raises(TypeError):
            solver.subhalo_mass_stripped(mass,3.,0.,method='pert2_shanks',unused=1)


def test_warm_particle_change_invalidates_host_and_endpoint_tables():
    import sashimi_w as w
    model=w.subhalos(2.)
    solver=w.TidalStrippingSolver(model,1e12,z_max=2.)
    old=solver.subhalo_mass_stripped(1e8,2.,0.,method='picard_table')
    model.mass_wdm=5.
    changed=solver.subhalo_mass_stripped(1e8,2.,0.,method='picard_table')
    fresh=w.TidalStrippingSolver(w.subhalos(5.),1e12,z_max=2.).subhalo_mass_stripped(1e8,2.,0.,method='picard_table')
    assert old!=changed
    np.testing.assert_array_equal(changed,fresh)


def test_long_solver_range_does_not_extend_an_in_domain_table():
    solver=AnalyticHost();solver.z_max=20.
    endpoint_mass(solver,1e6,3.,0.)
    assert next(iter(solver._picard_tables.values())).z_acc_max <= 7.+1e-13
    with pytest.warns(PicardFallbackWarning):
        endpoint_mass(solver,1e16,3.,0.)


def test_original_two_point_table_configuration_remains_supported():
    table=PicardTidalStrippingTable(AnalyticHost(),n_z_acc=2,n_log_ratio=2)
    assert np.isfinite(table.mass(1e6,3.))
    np.testing.assert_array_equal(table.mass([1.,2.],0.),[1.,2.])


def test_default_catalog_and_wrapper_forwarding_against_frozen_reference(monkeypatch):
    """Check the default Picard catalogue and public option forwarding.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped instrumentation of the dispatcher and returned catalogue.

    Notes
    -----
    Compare a compact 2 keV catalogue with independent author-source arrays
    using c=2.5 and the common top-hat amplitude. Pass with the existing
    1e-3 mass and 5e-3 other-column relative limits, exact survival, no
    fallback, and correct forwarding of explicit solver/profile options.
    This finite-grid regression is not a model calibration or total accuracy
    guarantee; the original arrays remain preserved under their old names.
    """
    import json
    import sashimi_w as module

    model = module.subhalos(2.)
    directory = Path(__file__).parent / 'data'
    config = json.loads((directory / 'catalogs-top-hat-c2.5.json').read_text())['cases']['B-q10-2.0']
    options = dict(config['parameters'])
    options.pop('method', None)
    original = module.TidalStrippingSolver.subhalo_mass_stripped
    selected_methods = []

    def record(self, *args, **kwargs):
        """Record the selected numerical method without changing its returned masses.

        Parameters
        ----------
        self : TidalStrippingSolver
            Original physical stripping provider.
        *args : tuple
            Physical masses in Msun and redshifts passed to the dispatcher.
        **kwargs : dict
            Solver choice and numerical controls forwarded unchanged.

        Returns
        -------
        numpy.ndarray
            Original final masses in Msun.

        Notes
        -----
        The enclosing test fails on a nondefault method or fallback.
        """
        selected_methods.append(kwargs.get('method'))
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module.TidalStrippingSolver, 'subhalo_mass_stripped', record)
        with warnings.catch_warnings():
            warnings.simplefilter('error', PicardFallbackWarning)
            actual = model.rs_rhos_calc(**options)
    assert set(selected_methods) == {'picard_table'}
    # Independently corrected frozen arrays retain their original checksums.
    with np.load(directory / config['file']) as expected:
        for i, value in enumerate(actual):
            assert np.all(np.isfinite(value))
            if i == 9:
                np.testing.assert_array_equal(value, expected[f'tuple_{i}'])
            else:
                np.testing.assert_allclose(value, expected[f'tuple_{i}'], atol=0,
                    rtol=1e-3 if i == 4 else 5e-3, err_msg=f'column {i}')

    def catalog(*args, **kwargs):
        """Supply the checked catalogue while verifying wrapper options.

        Parameters
        ----------
        *args : tuple
            Positional catalogue inputs supplied by the public wrapper.
        **kwargs : dict
            Requested solver, tolerance and profile-evolution choice.

        Returns
        -------
        tuple
            Previously checked legacy-unit WDM catalogue arrays.

        Notes
        -----
        Fail if explicit numerical/profile options are dropped; this helper
        does not simulate a new physical population.
        """
        assert kwargs['method'] == 'dop853'
        assert kwargs['rtol'] == 2e-10
        assert kwargs['profile_change'] is False
        return actual

    monkeypatch.setattr(model, 'rs_rhos_calc', catalog)
    for name, args in [('subhalo_distr', ()), ('N_sat', ()), ('N_sat_Vthres', (10.,))]:
        getattr(model, name)(options['M0'], *args, method='dop853',
                            rtol=2e-10, profile_change=False)
