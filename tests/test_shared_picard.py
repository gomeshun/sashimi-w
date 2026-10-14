"""WDM physical coefficients feed ITAMAE; the old ODE remains explicit."""

import numpy as np
import pytest
from itamae.evolution import PicardMassLossSolver
from sashimi_w import Subhalos
from sashimi_w._itamae_components import WDMHostHistory, WDMTidalMassLoss, WDMTidalSolver


@pytest.fixture(scope='module')
def solver():
    model = Subhalos(2.)
    law = WDMTidalMassLoss(model, WDMHostHistory(model, 1e12, 200, 0.))
    return WDMTidalSolver(law, z_min=0., z_max=3.)


def test_wdm_adapter_preserves_physical_rhs_and_matches_direct_reference(solver):
    assert isinstance(solver.backend, PicardMassLossSolver)
    masses = np.array([1e5, 1e8, 1e10])
    for z in [0., .7, 3.]:
        host = solver.reference_mass(z)
        rate = solver.rate(z) * masses * (masses / host)**solver.exponent(z)
        np.testing.assert_allclose(rate, solver.law.rhs(z, masses), rtol=4e-15)
    default = solver.mass(masses, 3., 0.)
    direct = solver.mass(masses, 3., 0., method='dop853')
    np.testing.assert_allclose(default, direct, rtol=1e-3)
    assert solver.backend.diagnostics()['fallback_calls'] == 0


def test_catalog_default_and_explicit_ode_have_identical_accretion_weights():
    parameters = dict(M0=1e10, dz=.5, zmax=1., N_ma=4, N_herm=2,
                      logmamin=6., logmamax=8., N_hermNa=3)
    default = Subhalos(2.).rs_rhos_catalog_calc(**parameters)
    direct = Subhalos(2.).rs_rhos_catalog_calc(**parameters, method='dop853')
    old = Subhalos(2.).rs_rhos_catalog_calc(**parameters, method='odeint')
    assert default.metadata['calculation_parameters']['method'] == 'picard_table'
    assert default.metadata['picard_diagnostics']['backend'] == 'itamae:picard-log-mass:v1'
    for name in ['weight_base', 'weight_concentration']:
        np.testing.assert_array_equal(default.weights[name], old.weights[name])
    for name in ['m200_acc', 'z_acc', 'r_s_acc', 'rho_s_acc']:
        np.testing.assert_array_equal(default.columns[name], old.columns[name])
    np.testing.assert_allclose(default.columns['m_bound'], direct.columns['m_bound'], rtol=1e-3)


def test_tuple_conversion_has_no_invented_solver_identity():
    parameters = dict(M0=1e10, dz=.5, zmax=1., N_ma=2, N_herm=1,
                      logmamin=6., logmamax=8., N_hermNa=3, method='odeint')
    model = Subhalos(2.)
    result = model.rs_rhos_calc(**parameters)
    converted = model.catalog_from_tuple(result)
    assert converted.metadata['solver_identifier'] == 'unknown:tuple-conversion'
    assert converted.metadata['picard_diagnostics'] is None
