"""Public observables count one survivor population, independent of histogram bins."""

import numpy as np
import pytest

from sashimi_w import Subhalos
from sashimi_w_physics import G, km, kpc, Msolar, pc, s


def population_model(survive):
    """Construct a synthetic population with known weights and a caller-selected survival mask.

    Parameters
    ----------
    survive : array_like of bool
        Controlled survival decisions for the synthetic weighted population.

    Returns
    -------
    tuple
        Toy model, accretion masses, expected-count weights, present speeds in km/s and
        captured calls.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    model = object.__new__(Subhalos)
    size = len(survive)
    row = np.arange(1, size + 1, dtype=float)
    mass = 10.0 ** (6.0 + row)
    weight = np.array([2.0, 3.0, 5.0])[:size]
    calls = []

    def population(*args, **kwargs):
        """Return the controlled historical-format catalogue without running a physical population
        model.

        Parameters
        ----------
        *args : tuple
            Additional positional parameters forwarded unchanged to the supplied numerical or
            physical callback.
        **kwargs : dict
            Additional keyword options forwarded to the called routine; reserved scientific
            controls remain owned by that routine.

        Returns
        -------
        tuple
            Ten aligned arrays with distinguishable accretion/current masses, weights and
            survival decisions.

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        calls.append(kwargs)
        return (
            mass.copy(),
            np.zeros(size),
            row[::-1].copy(),
            np.full(size, 0.1),
            mass / 10.0,
            row.copy(),
            np.full(size, 0.1),
            np.where(survive, 1.0, 0.5),
            weight.copy(),
            np.array(survive),
        )

    model.rs_rhos_calc = population
    current_velocity = (
        np.sqrt(4 * np.pi * G * (0.1 * (Msolar / pc**3)) / 4.625) * (row * kpc) / (km / s)
    )
    return model, mass, weight, current_velocity, calls


@pytest.mark.parametrize(
    "survive", [[True, True, True], [True, False, True], [False, False, False], [True], []]
)
@pytest.mark.parametrize("accretion", [True, False])
def test_mass_distribution_uses_survivors_in_both_mass_displays(survive, accretion):
    """Count surviving subhaloes whether the horizontal mass is accreted or current.

    Parameters
    ----------
    survive : sequence of bool
        Survival decisions for the controlled toy population.
    accretion : bool
        Whether the plotted mass is at accretion rather than after stripping; selection
        still uses survivors.

    Notes
    -----
    Pass criteria: For full, partial, empty and single-node populations, integrating the
    histogram recovers survivor weight and preserves the requested profile-change option.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, _, weight, _, calls = population_model(survive)
    mass, dndm = model.subhalo_distr(1e12, accretion=accretion, profile_change=False)
    reconstructed = np.sum(mass * dndm) * np.log(mass[1] / mass[0])
    assert reconstructed == pytest.approx(weight[np.array(survive, dtype=bool)].sum(), rel=1e-11)
    assert all(call["profile_change"] is False for call in calls)


@pytest.mark.parametrize(
    "survive", [[True, True, True], [True, False, True], [False, False, False], [True], []]
)
@pytest.mark.parametrize("threshold", [None, 1e8, 1e20])
def test_mass_threshold_total_and_exact_cumulative_axis(survive, threshold):
    """Apply the same strict peak-mass cut to total and cumulative surviving counts.

    Parameters
    ----------
    survive : sequence of bool
        Survival decisions for the controlled toy population.
    threshold : object
        Parametrized physical selection or half-mode threshold; the test states which values
        must be accepted or rejected.

    Notes
    -----
    Pass criteria: Totals and every cumulative point equal direct weighted selections for
    several thresholds and survival patterns; the axis is increasing.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, mass, weight, _, calls = population_model(survive)
    selected = np.array(survive, dtype=bool)
    if threshold is not None:
        selected &= mass > threshold
    total, x, counts = model.N_sat(
        1e12, Mpeak=threshold, Mpeak_thres=threshold is not None, profile_change=False
    )
    assert total == weight[selected].sum()
    expected = [weight[selected & (mass > edge)].sum() for edge in x]
    np.testing.assert_array_equal(counts, expected)
    assert len(x) == len(counts) == 10000
    assert np.all(np.diff(x) > 0)
    assert all(call["profile_change"] is False for call in calls)


@pytest.mark.parametrize(
    "survive", [[True, True, True], [True, False, True], [False, False, False], [True], []]
)
@pytest.mark.parametrize("peak", [True, False])
def test_velocity_selection_and_cumulative_current_velocity(survive, peak):
    """Distinguish peak-velocity selection from current-velocity cumulative counts.

    Parameters
    ----------
    survive : sequence of bool
        Survival decisions for the controlled toy population.
    peak : bool
        Whether the velocity threshold selects peak rather than current velocity.

    Notes
    -----
    Pass criteria: Totals use the requested peak/current threshold, cumulative values equal
    direct current-velocity selections and profile options are forwarded.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, _, weight, velocity, calls = population_model(survive)
    for threshold in [0.0, velocity[0] if len(velocity) else 1.0, 1e20]:
        selected = np.array(survive, dtype=bool)
        selected &= (velocity[::-1] if peak else velocity) > threshold
        total, x, counts = model.N_sat_Vthres(
            1e12, threshold, Vpeak_thres=peak, profile_change=False
        )
        assert total == weight[selected].sum()
        expected = [weight[selected & (velocity > edge)].sum() for edge in x]
        np.testing.assert_array_equal(counts, expected)
        assert len(x) == len(counts) == 10000
    assert all(call["profile_change"] is False for call in calls)
