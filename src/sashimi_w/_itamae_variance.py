"""Variance adapter for the SASHIMI-W ITAMAE migration.

The WDM transfer and WMAP7 spectrum belong to this variant. The corrected
Schneider mass assignment uses k_cut=2.5/R_th; EPS and concentration share
one top-hat sigma8 amplitude. ITAMAE supplies the continuous sharp-k integral.
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
from ._itamae_migration import ItamaeSubhalos, STANDARD_T2_Q10, CALCULATION_SPECIFICATION
from ._physics import Pk_file, WDM_TRANSFER_NU, h, k_file, sigma_8
from ._power import POWER_NORMALIZATION, SHARP_K_MASS_ASSIGNMENT


_SHARP_K_CUTOFF = SHARP_K_MASS_ASSIGNMENT


def _resolve_model(model, *, mass_wdm, wdm_power_convention):
    """Require a configured q10 thermal WDM model before constructing a variance provider.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    mass_wdm : float
        Positive thermal-relic WDM particle mass, numerical value in keV.
    wdm_power_convention : str, optional
        Supported standard-t2-q10 convention: the Viel transfer amplitude is squared in
        power. Historical published-q5 selectors are rejected.

    Returns
    -------
    tuple
        Resolved model and its supported power-convention identifier.

    Notes
    -----
    The current prescription uses the bundled spectrum converted to canonical Mpc units, a
    sharp-k mass-assignment factor 2.5, and one top-hat sigma8
    normalization. These are physical/convention choices, distinct from convergence of the
    integral. The tests pin the chosen prescription; they do not independently establish
    that it is the uniquely appropriate filter normalization.
    """
    if model is None:
        if wdm_power_convention is None:
            wdm_power_convention = STANDARD_T2_Q10
        model = ItamaeSubhalos(mass_wdm=mass_wdm, wdm_power_convention=wdm_power_convention)
    if wdm_power_convention == "published-q5":
        raise ValueError(
            "wdm_power_convention='published-q5' is retired; use the frozen reference."
        )
    selected = model.wdm_power_convention
    if selected != STANDARD_T2_Q10 or model.wdm_power_q != 10.0:
        raise ValueError("The supplied model must use the supported q10 thermal WDM specification.")
    if wdm_power_convention is not None and wdm_power_convention != selected:
        raise ValueError("wdm_power_convention does not match the configured model.")
    return model, selected


def make_variance_model(model=None, *, mass_wdm=1.5, wdm_power_convention=None):
    r"""Expose the standard continuous WDM variance through the common protocol.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    mass_wdm : float
        Positive thermal-relic WDM particle mass, numerical value in keV.
    wdm_power_convention : str, optional
        Supported standard-t2-q10 convention: the Viel transfer amplitude is squared in
        power. Historical published-q5 selectors are rejected.

    Returns
    -------
    CallableVarianceModel
        Public sigma and dS/dM callbacks of the configured continuous variance calculation.

    Notes
    -----
    The sharp-k mass assignment M=(4*pi/3)*rho*(c*R)**3 with c=2.5 follows Schneider. This
    calibrated mass mapping is distinct from the real-space top-hat radius used to define
    sigma8.

    The current prescription uses the bundled spectrum converted to canonical Mpc units, a
    sharp-k mass-assignment factor 2.5, and one top-hat sigma8
    normalization. These are physical/convention choices, distinct from convergence of the
    integral. The tests pin the chosen prescription; they do not independently establish
    that it is the uniquely appropriate filter normalization.

    References
    ----------

    .. [cite-sharp] Schneider (2015), Structure formation with suppressed small-scale perturbations, Eqs. (10)-(12).
        https://doi.org/10.1093/mnras/stv1169
        Full text: https://arxiv.org/abs/1412.2133
    """
    model, selected = _resolve_model(
        model, mass_wdm=mass_wdm, wdm_power_convention=wdm_power_convention
    )
    return CallableVarianceModel(
        identifier=f"{model.variance_model().identifier}:callable:v1",
        sigma_function=model.sigmaMz,
        derivative_function=model.dsdm,
    )


def make_integrated_variance_model(
    model: Any | None = None,
    *,
    mass_wdm: float = 1.5,
    wdm_power_convention: str | None = None,
    n_k: int = 4097,
) -> IntegratedVarianceModel:
    r"""Compose SASHIMI-W physics with ITAMAE's corrected sharp-k integrator.

    Parameters
    ----------
    model : object, optional
        Configured migrated WDM model. When omitted, a standard model is
        constructed from ``mass_wdm``.
    mass_wdm : float, optional
        WDM particle mass in keV used only when constructing a model.
    wdm_power_convention : {"standard-t2-q10"}, optional
        Defaults to the supported q10 thermal WDM specification. Retired q5
        selectors and models are explicitly rejected, without conversion.
    n_k : int
        Number of fixed logarithmic grid nodes across the configured spectrum bounds;
        spectrum knots also split cells and the final cutoff cell is integrated partially.

    Returns
    -------
    itamae.variance.IntegratedVarianceModel
        Continuous sharp-k variance with an analytic moving-boundary
        ``dS/dM`` supplied by ITAMAE.

    Notes
    -----
    The sharp-k mass assignment M=(4*pi/3)*rho*(c*R)**3 with c=2.5 follows Schneider. This
    calibrated mass mapping is distinct from the real-space top-hat radius used to define
    sigma8.

    The bundled k and P table is converted from h/Mpc and (Mpc/h)**3 to
    canonical Mpc**-1 and Mpc**3. Multiplication by h**2 converts the
    numerical mean density to Msun/Mpc**3. The Schneider mapping
    M=(4*pi/3)*rho*(2.5*R)**3 gives k_cut=2.5/R_th. The physical
    spectrum is normalized once under a top-hat of radius 8 Mpc/h;
    the same amplitude is used by the concentration calculation.
    In particular, sharp-k sigma(M8) is not forced to equal sigma8.

    The current prescription uses the bundled spectrum converted to canonical Mpc units, a
    sharp-k mass-assignment factor 2.5, and one top-hat sigma8
    normalization. These are physical/convention choices, distinct from convergence of the
    integral. The tests pin the chosen prescription; they do not independently establish
    that it is the uniquely appropriate filter normalization.

    References
    ----------

    .. [cite-sharp] Schneider (2015), Structure formation with suppressed small-scale perturbations, Eqs. (10)-(12).
        https://doi.org/10.1093/mnras/stv1169
        Full text: https://arxiv.org/abs/1412.2133
    """
    configured, selected_convention = _resolve_model(
        model,
        mass_wdm=mass_wdm,
        wdm_power_convention=wdm_power_convention,
    )
    hubble = float(h)
    canonical_k = np.asarray(k_file, dtype=float) * hubble
    canonical_power = np.asarray(Pk_file, dtype=float) / hubble**3
    canonical_rho_mean = float(configured.Rhomean_z) * hubble**2
    canonical_alpha = float(configured.a) / hubble
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
        r"""Evaluate a snapshot of the configured thermal suppression in canonical wavenumber units.

        Parameters
        ----------
        wavenumber : array_like
            Canonical Fourier wavenumber in Mpc^-1.

        Returns
        -------
        numpy.ndarray
            Dimensionless q10 power ratio with alpha fixed when the provider is constructed.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        The current prescription uses the bundled spectrum converted to canonical Mpc units, a
        sharp-k mass-assignment factor 2.5, and one top-hat sigma8
        normalization. These are physical/convention choices, distinct from convergence of the
        integral. The tests pin the chosen prescription; they do not independently establish
        that it is the uniquely appropriate filter normalization.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        k = np.asarray(wavenumber, dtype=float)
        # Snapshot the configured alpha and q rather than retaining a mutable
        # model callback behind an already calculated content identifier.
        return (1.0 + (canonical_alpha * k) ** (2.0 * WDM_TRANSFER_NU)) ** (-q / WDM_TRANSFER_NU)

    particle_mass = float(configured.mass_wdm)
    integration_options = {
        "window": SharpKWindow(),
        "rho_mean": canonical_rho_mean,
        "k_min": float(canonical_k[0]),
        "k_max": float(canonical_k[-1]),
        "n_k": n_k,
        "filter_scale": _SHARP_K_CUTOFF,
    }
    normalization = configured._power_normalization()

    def normalized_wdm_power_ratio(wavenumber: Any) -> np.ndarray:
        r"""Apply the common top-hat sigma8 normalization to the thermal WDM power ratio.

        Parameters
        ----------
        wavenumber : array_like
            Canonical Fourier wavenumber in Mpc^-1.

        Returns
        -------
        numpy.ndarray
            Power suppression divided by normalization**2, with the wavenumber shape.

        Notes
        -----
        Viel et al. give the thermal-relic amplitude T = [1+(alpha*k)**(2*nu)]**(-5/nu),
        nu=1.12, and alpha=0.049*(m/keV)**(-1.11)*(Omega_WDM/0.25)**0.11*(h/0.7)**1.22 in Mpc/h.
        Power uses T**2, hence exponent -10/nu; k is in h/Mpc.

        The current prescription uses the bundled spectrum converted to canonical Mpc units, a
        sharp-k mass-assignment factor 2.5, and one top-hat sigma8
        normalization. These are physical/convention choices, distinct from convergence of the
        integral. The tests pin the chosen prescription; they do not independently establish
        that it is the uniquely appropriate filter normalization.

        References
        ----------

        .. [cite-viel] Viel et al. (2005), Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest, Eqs. (6)-(7).
            https://arxiv.org/abs/astro-ph/0501562
        """
        return wdm_power_ratio(wavenumber) / normalization**2

    power = TransferModifiedPowerSpectrum(
        base,
        normalized_wdm_power_ratio,
        ratio_identifier=(
            f"{CALCULATION_SPECIFICATION}:wdm-power-ratio:"
            f"m_wdm_keV={particle_mass:.17g};"
            f"alpha_Mpc={canonical_alpha:.17g};"
            f"nu={WDM_TRANSFER_NU:.17g};"
            f"convention={selected_convention};"
            f"formula-role={formula_role};q={q:.17g};"
            f"half-mode={half_mode_definition};"
            f"normalization={POWER_NORMALIZATION};sigma8-target={float(sigma_8):.17g};"
            f"sigma8-normalization={normalization:.17g}"
        ),
    )
    return IntegratedVarianceModel(
        power=power,
        **integration_options,
        growth_function=configured.growthD,
        growth_identifier=(
            f"growth={configured.itamae_cosmology.identifier};calculation=cpt-normalized-v1"
        ),
    )


def _sharp_k_variance_gap(variance, smaller_mass, larger_mass):
    """Integrate a variance interval without subtracting saturated variances.

    The pinned ITAMAE exposes no interval API. Use its public configuration,
    normalized canonical spectrum, fixed log cells, spectrum knots and Gauss
    order unchanged. This helper supplies only the EPS gap at redshift zero;
    standard sigma and derivative evaluation still belong to ITAMAE.

    Parameters
    ----------
    variance : IntegratedVarianceModel
        Configured canonical sharp-k provider supplying the spectrum, density, fixed-cell
        quadrature and mass assignment.
    smaller_mass : float or array_like
        Positive ordered halo masses in physical Msun; smaller_mass <= larger_mass defines a
        nonnegative sharp-k variance interval.
    larger_mass : float or array_like
        Positive ordered halo masses in physical Msun; smaller_mass <= larger_mass defines a
        nonnegative sharp-k variance interval.

    Returns
    -------
    numpy.ndarray
        Nonnegative dimensionless variance interval with broadcast mass shape, including
        small positive suppressed-tail contributions.

    Notes
    -----
    The current prescription uses the bundled spectrum converted to canonical Mpc units, a
    sharp-k mass-assignment factor 2.5, and one top-hat sigma8
    normalization. These are physical/convention choices, distinct from convergence of the
    integral. The tests pin the chosen prescription; they do not independently establish
    that it is the uniquely appropriate filter normalization.
    """
    small, large = np.broadcast_arrays(
        np.asarray(smaller_mass, dtype=float), np.asarray(larger_mass, dtype=float)
    )
    if (
        not np.all(np.isfinite(small))
        or not np.all(np.isfinite(large))
        or np.any(small <= 0)
        or np.any(small > large)
    ):
        raise ValueError("Variance gap requires finite positive ordered masses.")
    if not isinstance(variance.window, SharpKWindow):
        raise TypeError("WDM EPS variance gaps require a sharp-k window.")
    if not small.size:
        return np.empty(small.shape)
    edges = np.linspace(np.log(variance.k_min), np.log(variance.k_max), variance.n_k)
    knots = np.asarray(variance.power.integration_breakpoints)
    interior = knots[(knots > variance.k_min) & (knots < variance.k_max)]
    edges = np.unique(np.concatenate((edges, np.log(interior))))
    nodes, weights = np.polynomial.legendre.leggauss(variance.sharp_k_order)
    fractions, weights = (nodes + 1) / 2, weights / 2

    def integrate(left, right):
        """Integrate complete or partial log-k cells without subtracting large saturated variances.

        Parameters
        ----------
        left : array_like
            Lower log(k/Mpc^-1) edge of each interval.
        right : array_like
            Upper log(k/Mpc^-1) edge of each interval.

        Returns
        -------
        numpy.ndarray
            One dimensionless variance contribution per flattened interval.

        Notes
        -----
        The current prescription uses the bundled spectrum converted to canonical Mpc units, a
        sharp-k mass-assignment factor 2.5, and one top-hat sigma8
        normalization. These are physical/convention choices, distinct from convergence of the
        integral. The tests pin the chosen prescription; they do not independently establish
        that it is the uniquely appropriate filter normalization.
        """
        left, right = np.broadcast_arrays(left, right)
        result = np.empty(left.size)
        left, right = left.reshape(-1), right.reshape(-1)
        for start in range(0, result.size, variance.chunk_size):
            stop = start + variance.chunk_size
            width = right[start:stop] - left[start:stop]
            logk = left[start:stop, None] + width[:, None] * fractions
            k = np.clip(np.exp(logk), variance.k_min, variance.k_max)
            power = np.asarray(variance.power(k))
            if power.shape != k.shape or not np.all(np.isfinite(power)) or np.any(power < 0):
                raise ValueError("Power spectrum must return aligned finite nonnegative values.")
            result[start:stop] = width * np.sum(k**3 * power * weights / (2 * np.pi**2), axis=1)
        return result

    def cutoff(mass):
        """Map halo mass to the clipped logarithmic sharp-k cutoff.

        Parameters
        ----------
        mass : float or array_like
            Positive halo mass. Use solar masses with the canonical backend; a supplied variance
            prescription must use the same mass convention and factors of h.

        Returns
        -------
        numpy.ndarray
            ln(k_c) with k_c clipped to the configured canonical spectrum bounds.

        Notes
        -----
        The current prescription uses the bundled spectrum converted to canonical Mpc units, a
        sharp-k mass-assignment factor 2.5, and one top-hat sigma8
        normalization. These are physical/convention choices, distinct from convergence of the
        integral. The tests pin the chosen prescription; they do not independently establish
        that it is the uniquely appropriate filter normalization.
        """
        radius = (3 * mass / (4 * np.pi * variance.rho_mean)) ** (1 / 3)
        return np.log(np.clip(variance.filter_scale / radius, variance.k_min, variance.k_max))

    complete = integrate(edges[:-1], edges[1:])
    cumulative = np.r_[0.0, np.cumsum(complete)]
    tail = np.r_[np.cumsum(complete[::-1])[::-1], 0.0]
    left, right = cutoff(large).reshape(-1), cutoff(small).reshape(-1)
    i = np.clip(np.searchsorted(edges, left, side="right") - 1, 0, len(edges) - 2)
    j = np.clip(np.searchsorted(edges, right, side="right") - 1, 0, len(edges) - 2)
    result = np.empty(left.shape)
    same = i == j
    result[same] = integrate(left[same], right[same])
    a, b = i[~same], j[~same]
    # Choose forward sums in the rising spectrum and reverse sums in the
    # suppressed tail, so positive high-k support survives variance saturation.
    middle = np.where(
        cumulative[b] < tail[a + 1],
        cumulative[b] - cumulative[a + 1],
        tail[a + 1] - tail[b],
    )
    result[~same] = (
        integrate(left[~same], edges[a + 1]) + middle + integrate(edges[b], right[~same])
    )
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("Variance gap integration produced invalid values.")
    return result.reshape(small.shape)


__all__ = ["make_integrated_variance_model", "make_variance_model"]
