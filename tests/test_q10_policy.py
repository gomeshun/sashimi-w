"""The adopted thermal WDM formula and old-input rejection contract."""

import numpy as np
import pytest
import sashimi_w
from sashimi_w import Subhalos
from sashimi_w_itamae_variance import make_integrated_variance_model, make_variance_model


@pytest.mark.parametrize("factory", [Subhalos, make_integrated_variance_model, make_variance_model])
def test_retired_q5_is_rejected_instead_of_reinterpreted(factory):
    with pytest.raises(ValueError, match="retired"):
        factory(wdm_power_convention="published-q5")


def test_public_default_is_q10_and_old_constant_is_removed():
    model = Subhalos(2.0)
    assert model.wdm_power_q == 10
    assert model.wdm_power_convention == "standard-t2-q10"
    assert not hasattr(sashimi_w, "PUBLISHED_Q5")
    for field in ["wdm_power_q", "wdm_power_convention", "half_mode_power_ratio"]:
        with pytest.raises(AttributeError):
            setattr(model, field, 5)


def test_transfer_and_power_are_distinct_and_both_half_modes_are_explicit():
    model = Subhalos(2.0, wdm_power_convention="standard-t2-q10")
    k = np.geomspace(1e-4, 1e3, 101).reshape(101, 1)
    np.testing.assert_allclose(model.power_ratio(k), model.transfer_amplitude(k) ** 2, rtol=5e-15)
    assert model.transfer_amplitude(0) == model.power_ratio(0) == 1
    amplitude_half = model.half_mode_wavenumber()
    assert model.transfer_amplitude(amplitude_half) == pytest.approx(0.5, rel=2e-15)
    assert model.power_ratio(amplitude_half) == pytest.approx(0.25, rel=2e-15)
    power_half = model.half_mode_wavenumber(power_ratio=0.5)
    assert model.power_ratio(power_half) == pytest.approx(0.5, rel=2e-15)
    assert power_half < amplitude_half
    np.testing.assert_array_equal(model._wdm_power_ratio(k), model.power_ratio(k))


@pytest.mark.parametrize("threshold", [0, 1, -1, np.nan, np.inf, [0.5], True])
def test_half_mode_threshold_must_be_a_real_scalar_between_zero_and_one(threshold):
    model = Subhalos(2, wdm_power_convention="standard-t2-q10")
    with pytest.raises(ValueError, match="power_ratio"):
        model.half_mode_wavenumber(power_ratio=threshold)
