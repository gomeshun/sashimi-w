"""Exact variance-gap and mass-axis checks for WDM EPS normalization reuse."""

import numpy as np
import pytest

from sashimi_w import subhalos


@pytest.mark.parametrize("mass_wdm", [0.5, 2.0, 5.0])
def test_normalization_reuse_preserves_mass_axis(monkeypatch, mass_wdm):
    """Remove repeated normalization integrals without changing their values.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restores the observed variance kernel after this controlled calculation.
    mass_wdm : float
        Thermal relic mass in keV, spanning strongly and weakly suppressed power.

    Notes
    -----
    At four epochs, a mixed mass grid includes forbidden EPS support. Each
    observed gap batch is checked byte-for-byte against independently evaluated
    repeated copies of its inputs. The normalization batch must contain at most
    one interval per host-history/epoch pair, including high-order tails.
    Permuting the subhalo mass axis must only permute the output, and a grid
    wholly outside support must yield zeros with an empty normalization batch.
    These are numerical identity and work-reduction checks, not calibration of
    the EPS prescription or equality across distinct library versions.
    """
    model = subhalos(mass_wdm)
    owner, name = model._variance, "gap"
    original = getattr(owner, name)
    sizes = []

    def observe(smaller, larger):
        """Check that batch compaction preserves each physical variance interval.

        Parameters
        ----------
        smaller, larger : numpy.ndarray
            Ordered halo masses in physical Msun with the same one-dimensional shape.

        Returns
        -------
        numpy.ndarray
            The original dimensionless sharp-k variance gaps, unchanged.

        Notes
        -----
        Repeating each interval three times changes batching without changing any
        integration endpoints. Byte equality detects altered quadrature arithmetic;
        it does not independently validate the power spectrum.
        """
        result = original(smaller, larger)
        repeated = original(np.repeat(smaller, 3), np.repeat(larger, 3)).reshape(-1, 3)
        assert np.repeat(result, 3).tobytes() == repeated.tobytes()
        sizes.append(np.size(smaller))
        return result

    monkeypatch.setattr(owner, name, observe)
    epochs = np.array([.025, 1., 3., 5.])
    masses = np.broadcast_to(np.geomspace(1e5, 1e13, 61), (4, 61)).copy()
    result = model.Na_calc(masses, epochs, 1e12, N_herm=20)
    assert np.all(np.isfinite(result))
    assert sizes[1] <= 20 * len(epochs) < sizes[0]
    order = np.random.default_rng(42).permutation(masses.shape[-1])
    permuted = model.Na_calc(masses[:, order], epochs, 1e12, N_herm=20)
    assert permuted.tobytes() == result[:, order].tobytes()
    empty = model.Na_calc(np.full((4, 3), 1e15), epochs, 1e12, N_herm=20)
    assert np.count_nonzero(empty) == 0
    assert sizes[-1] == 0
