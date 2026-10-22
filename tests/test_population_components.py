"""Physical node transport checks independent of the old population loop."""

from types import SimpleNamespace

import numpy as np
from itamae.execution import PopulationComponents
from itamae.halo import invert_nfw_mass_function, nfw_mass_function
from itamae.measure import build_accretion_batch
from sashimi_w import kpc
from sashimi_w_itamae_components import (
    WDMCatalogColumns,
    WDMInitialStructure,
    WDMProfileEvolution,
    WDMSurvival,
)


def test_canonical_transport_matches_half_mass_solution_and_chunking():
    """Transport an analytic half-mass WDM evolution through canonical structure stages.

    Notes
    -----
    Pass criteria: The bound mass is half the input to 1e-7, radii use the correct units,
    NFW mass reconstruction agrees to 3e-13, and repeated batches preserve all node values.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = SimpleNamespace(
        fc=nfw_mass_function, _invert_nfw_mass_fraction=invert_nfw_mass_function
    )
    mass = np.array([1e6, 1e7])
    batch = build_accretion_batch(mass, 1.0, [5.0, 8.0], [3.0, 7.0], [0.5, 0.5], mvir_acc=mass)
    context = {"rvir_cgs": np.array([1e-4, 2e-4]) * 1000.0 * kpc, "ma": mass, "z_acc": 1.0}
    law = SimpleNamespace(rhs=lambda z, m: np.log(2.0) * m)
    components = PopulationComponents(
        WDMInitialStructure(model, 1),
        WDMProfileEvolution(model, law, 0.0, 1, False, method="odeint"),
        WDMSurvival(0.0),
        WDMCatalogColumns(),
    )
    result = components.execute([batch], contexts=[context])
    np.testing.assert_allclose(result.columns["m_bound"], mass / 2.0, rtol=1e-7)
    np.testing.assert_allclose(result.columns["r_s_acc"], np.array([1e-4, 2e-4]) / [5.0, 8.0])
    reconstructed = (
        4
        * np.pi
        * result.columns["rho_s"]
        * result.columns["r_s"] ** 3
        * nfw_mass_function(result.columns["c_t"])
    )
    np.testing.assert_allclose(reconstructed, result.columns["m_bound"], rtol=3e-13)
    np.testing.assert_array_equal(result.batch.m200_acc, mass)
    np.testing.assert_array_equal(result.batch.weight_base, [3.0, 7.0])
    repeated = components.execute([batch, batch], contexts=[context, context])
    for name in result.columns:
        np.testing.assert_array_equal(repeated.columns[name], np.tile(result.columns[name], 2))
