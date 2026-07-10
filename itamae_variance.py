"""Variance adapter for the SASHIMI-W ITAMAE migration.

The WDM transfer function, tabulated WMAP7 spectrum, sharp-k filtering,
interpolation tables, and concentration calibration remain in ``sashimi_w.py``.
This module exposes the resulting mass variance through ITAMAE's common
interface without changing those physical choices.
"""

from __future__ import annotations

from typing import Any

from itamae.variance import CallableVarianceModel
from itamae_migration import ItamaeSubhalos


def make_variance_model(model: Any | None = None, *, mass_wdm: float = 1.5) -> CallableVarianceModel:
    """Wrap the configured SASHIMI-W sharp-k variance implementation.

    Parameters
    ----------
    model : object, optional
        Existing SASHIMI-W model. A migrated model is constructed when omitted.
    mass_wdm : float, optional
        WDM particle mass in keV used only when constructing a model.

    Returns
    -------
    itamae.variance.CallableVarianceModel
        Variance model backed by SASHIMI-W ``sigmaMz`` and ``dsdm``.
    """

    model = model or ItamaeSubhalos(mass_wdm=mass_wdm)
    particle_mass = float(model.mass_wdm)
    return CallableVarianceModel(
        identifier=f"sashimi-w:m_wdm_keV={particle_mass}:sharp-k:v1",
        sigma_function=lambda mass, z: model.sigmaMz(mass, z),
        derivative_function=lambda mass, z: model.dsdm(mass, z),
    )


__all__ = ["make_variance_model"]
