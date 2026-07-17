"""Variance adapter for the SASHIMI-W ITAMAE migration.

The WDM transfer function, tabulated WMAP7 spectrum, sharp-k filtering,
interpolation tables, and concentration calibration remain in ``sashimi_w.py``.
This module exposes the resulting mass variance through ITAMAE's common
interface without changing those physical choices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from itamae.power import (
    SharpKWindow,
    TabulatedPowerSpectrum,
    TransferModifiedPowerSpectrum,
)
from itamae.variance import CallableVarianceModel, IntegratedVarianceModel
from sashimi_w_itamae_migration import ItamaeSubhalos, PUBLISHED_Q5
from sashimi_w import Pk_file, WDM_TRANSFER_NU, h, k_file, sigma_8


_SHARP_K_MASS_ASSIGNMENT = 2.5
_SHARP_K_CUTOFF = _SHARP_K_MASS_ASSIGNMENT * (9.0 * np.pi / 2.0) ** (1.0 / 3.0)


def _resolve_model(
    model: Any | None,
    *,
    mass_wdm: float,
    physics_mode: str,
    wdm_power_convention: str | None,
) -> tuple[Any, str]:
    """Resolve one convention without inferring it from ``physics_mode``."""
    if model is None:
        if wdm_power_convention is None:
            raise ValueError(
                "wdm_power_convention must be explicit when constructing an ITAMAE model."
            )
        configured = ItamaeSubhalos(
            mass_wdm=mass_wdm,
            physics_mode=physics_mode,
            wdm_power_convention=wdm_power_convention,
        )
        return configured, wdm_power_convention

    selected = getattr(model, "wdm_power_convention", PUBLISHED_Q5)
    if wdm_power_convention is not None and wdm_power_convention != selected:
        raise ValueError(
            "Requested wdm_power_convention does not match the configured model: "
            f"{wdm_power_convention!r} != {selected!r}."
        )
    return model, selected


def make_variance_model(
    model: Any | None = None,
    *,
    mass_wdm: float = 1.5,
    physics_mode: str = "consistent",
    wdm_power_convention: str | None = None,
) -> CallableVarianceModel:
    """Wrap the configured SASHIMI-W sharp-k variance implementation.

    Parameters
    ----------
    model : object, optional
        Existing SASHIMI-W model. A migrated model is constructed when omitted.
    mass_wdm : float, optional
        WDM particle mass in keV used only when constructing a model.
    physics_mode : {"consistent", "legacy"}, optional
        Growth and derivative convention used only when constructing a model.
    wdm_power_convention : {"published-q5", "standard-t2-q10"}, optional
        Required when constructing a migrated model. If ``model`` is supplied,
        an explicitly supplied value must match it. A plain public
        ``sashimi_w.subhalos`` model is identified as published q5.

    Returns
    -------
    itamae.variance.CallableVarianceModel
        Variance model backed by SASHIMI-W ``sigmaMz`` and ``dsdm``.
    """

    model, selected_convention = _resolve_model(
        model,
        mass_wdm=mass_wdm,
        physics_mode=physics_mode,
        wdm_power_convention=wdm_power_convention,
    )
    particle_mass = float(model.mass_wdm)
    selected_mode = getattr(model, "physics_mode", "legacy")
    q = float(getattr(model, "wdm_power_q", 5.0))
    formula_role = getattr(
        model,
        "wdm_power_formula_role",
        "q5-expression-used-as-power-ratio",
    )
    half_mode_definition = getattr(
        model,
        "half_mode_definition",
        "P_WDM/P_CDM=0.5",
    )
    return CallableVarianceModel(
        identifier=(
            f"sashimi-w:m_wdm_keV={particle_mass:.17g}:sharp-k:"
            f"physics={selected_mode}:power={selected_convention}:"
            f"formula-role={formula_role}:q={q:.17g}:"
            f"half-mode={half_mode_definition}:legacy-callable:v3"
        ),
        sigma_function=lambda mass, z: model.sigmaMz(mass, z),
        derivative_function=lambda mass, z: model.dsdm(mass, z),
    )


def make_integrated_variance_model(
    model: Any | None = None,
    *,
    mass_wdm: float = 1.5,
    wdm_power_convention: str | None = None,
    n_k: int = 4097,
) -> IntegratedVarianceModel:
    """Compose SASHIMI-W physics with ITAMAE's corrected sharp-k integrator.

    Parameters
    ----------
    model : object, optional
        Configured migrated WDM model. When omitted, a consistent-mode model is
        constructed from ``mass_wdm``.
    mass_wdm : float, optional
        WDM particle mass in keV used only when constructing a model.
    wdm_power_convention : {"published-q5", "standard-t2-q10"}, optional
        Required when constructing a model. If ``model`` is supplied, an
        explicitly supplied value must match the model rather than replacing
        it. This prevents a q5 top-hat concentration from being paired with a
        q10 sharp-k variance, or vice versa.
    n_k : int, optional
        Number of logarithmic integration nodes between the bundled spectrum
        endpoints and each mass-dependent sharp-k cutoff.

    Returns
    -------
    itamae.variance.IntegratedVarianceModel
        Continuous sharp-k variance with an analytic moving-boundary
        ``dS/dM`` supplied by ITAMAE.

    Notes
    -----
    The bundled table uses ``k`` in ``h/Mpc`` and power in ``(Mpc/h)^3``.
    This opt-in path converts it to canonical ``1/Mpc`` and ``Mpc^3`` before
    constructing the ITAMAE model. The mean density is likewise converted from
    the legacy numerical ``(Msun/h)/(Mpc/h)^3`` convention to ``Msun/Mpc^3``.
    SASHIMI-W defines ``R=R_th/2.5`` and integrates to
    ``k_c=(9*pi/2)^(1/3)/R``. ITAMAE's ``filter_scale`` therefore receives the
    combined coefficient ``2.5*(9*pi/2)^(1/3)``, not merely 2.5.
    """
    configured, selected_convention = _resolve_model(
        model,
        mass_wdm=mass_wdm,
        physics_mode="consistent",
        wdm_power_convention=wdm_power_convention,
    )
    if getattr(configured, "physics_mode", None) != "consistent":
        raise ValueError(
            "IntegratedVarianceModel enforces S=sigma**2 and requires "
            "physics_mode='consistent'; use make_variance_model for legacy "
            "reproduction."
        )
    hubble = float(h)
    canonical_k = np.asarray(k_file, dtype=float) * hubble
    canonical_power = np.asarray(Pk_file, dtype=float) / hubble**3
    canonical_rho_mean = float(configured.Rhomean_z) * hubble**2
    canonical_alpha = float(configured.a) / hubble
    sigma8_mass = float(configured.MassIn8Mpc) / hubble
    q = float(configured.wdm_power_q)
    formula_role = str(configured.wdm_power_formula_role)
    half_mode_definition = str(configured.half_mode_definition)

    base = TabulatedPowerSpectrum(
        canonical_k,
        canonical_power,
        identifier=(
            "sashimi-w:WMAP7-camb:z=0;"
            "source-k-unit=h/Mpc;source-power-unit=(Mpc/h)^3;"
            "unit-conversion=canonical-Mpc"
        ),
        interpolation="linear",
    )

    def wdm_power_ratio(wavenumber: Any) -> np.ndarray:
        k = np.asarray(wavenumber, dtype=float)
        return np.asarray(configured._wdm_power_ratio(k, alpha=canonical_alpha), dtype=float)

    particle_mass = float(configured.mass_wdm)
    raw_power = TransferModifiedPowerSpectrum(
        base,
        wdm_power_ratio,
        ratio_identifier=(
            "sashimi-w:wdm-power-ratio:unnormalized:"
            f"m_wdm_keV={particle_mass:.17g};"
            f"alpha_Mpc={canonical_alpha:.17g};"
            f"nu={WDM_TRANSFER_NU:.17g};"
            f"convention={selected_convention};"
            f"formula-role={formula_role};q={q:.17g};"
            f"half-mode={half_mode_definition}"
        ),
    )
    integration_options = {
        "window": SharpKWindow(),
        "rho_mean": canonical_rho_mean,
        "k_min": float(canonical_k[0]),
        "k_max": float(canonical_k[-1]),
        "n_k": n_k,
        "filter_scale": _SHARP_K_CUTOFF,
    }
    raw_variance = IntegratedVarianceModel(
        power=raw_power,
        **integration_options,
    )
    raw_sigma8 = float(raw_variance.sigma(sigma8_mass, 0.0))
    normalization = raw_sigma8 / float(sigma_8)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("The ITAMAE-integrated sigma8 normalization must be positive.")

    def normalized_wdm_power_ratio(wavenumber: Any) -> np.ndarray:
        return wdm_power_ratio(wavenumber) / normalization**2

    power = TransferModifiedPowerSpectrum(
        base,
        normalized_wdm_power_ratio,
        ratio_identifier=(
            "sashimi-w:wdm-power-ratio:"
            f"m_wdm_keV={particle_mass:.17g};"
            f"alpha_Mpc={canonical_alpha:.17g};"
            f"nu={WDM_TRANSFER_NU:.17g};"
            f"convention={selected_convention};"
            f"formula-role={formula_role};q={q:.17g};"
            f"half-mode={half_mode_definition};"
            f"sigma8-normalization={normalization:.17g}"
        ),
    )
    return IntegratedVarianceModel(
        power=power,
        **integration_options,
        growth_function=configured.growthD,
        growth_identifier=(f"growth={configured.itamae_cosmology.identifier};physics=consistent"),
    )


__all__ = ["make_integrated_variance_model", "make_variance_model"]
