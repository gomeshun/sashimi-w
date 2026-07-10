"""ITAMAE compatibility layer for incremental SASHIMI-W migration.

SASHIMI-W differs materially from the C/SI/F implementations: it uses CGS
module constants, WMAP7 parameters loaded from a power-spectrum file, a sharp-k
variance calculation, and a distinct concentration prescription. This first
adapter therefore replaces only the linear-growth calculation. WDM transfer,
window, variance, and concentration code remain legacy-controlled until their
own golden regression fixtures are established.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from itamae.cosmology import NativeFlatLCDM
from sashimi_w import OmegaM, h, subhalos


class ItamaeSubhalos(subhalos):
    """SASHIMI-W population model with an ITAMAE growth-factor backend.

    Parameters
    ----------
    mass_wdm : float, optional
        Warm-dark-matter particle mass in keV, following the legacy API.
    cosmology_backend : object, optional
        ITAMAE-compatible cosmology backend. The default backend uses the WMAP7
        matter density and reduced Hubble parameter read by ``sashimi_w``.

    Notes
    -----
    The legacy ``linear_growth_factor`` method has an unusual two-element input
    convention: a two-element redshift array represents ``(z1, z2)`` and returns
    the ratio ``D(z2) / D(z1)``, whereas arrays of any other size are interpreted
    as redshifts relative to zero. This adapter preserves that behavior exactly.

    No unit conversion is performed here because the growth factor is
    dimensionless. The CGS-to-ITAMAE unit boundary for masses, lengths, and
    densities will be introduced only after WDM variance and concentration
    regression fixtures are available.
    """

    def __init__(self, mass_wdm: float = 1.5, cosmology_backend: Any | None = None) -> None:
        super().__init__(mass_wdm=mass_wdm)
        self.itamae_cosmology = cosmology_backend or NativeFlatLCDM(
            omega_m0=float(OmegaM), h=float(h)
        )

    def linear_growth_factor(self, Omega_m0: float, Omega_l0: float, z: Any):
        """Return the legacy-compatible linear growth factor using ITAMAE.

        Parameters
        ----------
        Omega_m0 : float
            Present-day matter density fraction.
        Omega_l0 : float
            Present-day cosmological-constant density fraction. SASHIMI-W uses a
            flat model in the calls migrated here, so this value is validated
            against ``1 - Omega_m0``.
        z : float or numpy.ndarray
            Redshift. A two-element array is interpreted as an interval
            ``(z1, z2)``; otherwise the result is normalized at redshift zero.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless growth factor or interval growth ratio.

        Raises
        ------
        ValueError
            If the supplied density fractions are not consistent with a flat
            matter-plus-lambda cosmology.
        """

        if not np.isclose(Omega_m0 + Omega_l0, 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("The initial ITAMAE WDM adapter supports flat cosmologies only.")

        backend = self.itamae_cosmology
        backend_omega_m = getattr(backend, "omega_m0", Omega_m0)
        if not np.isclose(backend_omega_m, Omega_m0, rtol=0.0, atol=1.0e-12):
            backend = NativeFlatLCDM(omega_m0=float(Omega_m0), h=float(h))

        redshift = np.asarray(z, dtype=float)
        flattened = np.atleast_1d(redshift)
        if flattened.size == 2:
            z1, z2 = flattened[0], flattened[1]
            return np.asarray(backend.growth_factor(z2)) / np.asarray(
                backend.growth_factor(z1)
            )

        result = np.asarray(backend.growth_factor(redshift)) / np.asarray(
            backend.growth_factor(0.0)
        )
        return float(result) if result.ndim == 0 else result


__all__ = ["ItamaeSubhalos"]
