"""Incremental ITAMAE migration layer for SASHIMI-W.

SASHIMI-W differs materially from the C/SI/F implementations: it uses CGS
module constants, WMAP7 parameters loaded from a tabulated power spectrum, a
sharp-k variance calculation, and a distinct concentration prescription. This
adapter therefore keeps all WDM population physics in ``sashimi_w.py`` while
migrating the growth-factor boundary, NFW inversion, unit conversion, and
weighted-catalog representation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from itamae.backends import BackendConfig
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import invert_nfw_mass_function
from itamae.types import CATALOG_SCHEMA_VERSION, WeightedSubhaloCatalog
from itamae.units import NativeUnits
import sashimi_w as _legacy
from sashimi_w import OmegaM, h, subhalos


_CANONICAL_SCALE = {
    "Msun": 1.0,
    "kpc": 1.0e-3,
    "Msun/pc3": 1.0e18,
    "dimensionless": 1.0,
}

_CANONICAL_ASTROPY_UNIT = {
    "mass": "Msun",
    "length": "Mpc",
    "density": "Msun / Mpc3",
}


class ItamaeSubhalos(subhalos):
    """SASHIMI-W model with explicit ITAMAE cosmology and unit boundaries.

    Parameters
    ----------
    mass_wdm : float, optional
        Warm-dark-matter particle mass in keV, following the legacy API.
    cosmology_backend : object, optional
        ITAMAE-compatible cosmology backend. The default uses the WMAP7 matter
        density and reduced Hubble parameter read by ``sashimi_w``.
    unit_backend : object, optional
        ITAMAE-compatible unit backend. Native canonical floating-point units
        are used by default.

    Notes
    -----
    The legacy ``linear_growth_factor`` method has an unusual two-element input
    convention: a two-element redshift array represents ``(z1, z2)`` and returns
    ``D(z2) / D(z1)``, whereas arrays of any other size are normalized at zero.
    This adapter preserves that behavior.

    Legacy catalog output mixes solar masses, kiloparsecs, and solar masses per
    cubic parsec. ``rs_rhos_catalog_calc`` converts them to ITAMAE canonical
    units: solar masses, megaparsecs, and solar masses per cubic megaparsec.
    """

    def __init__(
        self,
        mass_wdm: float = 1.5,
        cosmology_backend: Any | None = None,
        unit_backend: Any | None = None,
    ) -> None:
        super().__init__(mass_wdm=mass_wdm)
        self.itamae_cosmology = cosmology_backend or NativeFlatLCDM(
            omega_m0=float(OmegaM), h=float(h)
        )
        self.itamae_units = unit_backend or NativeUnits()

    def linear_growth_factor(self, Omega_m0: float, Omega_l0: float, z: Any):
        """Return the legacy-compatible linear growth factor using ITAMAE.

        Parameters
        ----------
        Omega_m0 : float
            Present-day matter density fraction.
        Omega_l0 : float
            Present-day cosmological-constant density fraction.
        z : float or numpy.ndarray
            Redshift. A two-element array is interpreted as an interval.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless growth factor or interval growth ratio.
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

    def _to_canonical(self, value, physical_type: str, legacy_unit: str) -> np.ndarray:
        """Convert a legacy WDM array to the active ITAMAE unit backend.

        Parameters
        ----------
        value
            Legacy scalar or array.
        physical_type : str
            ITAMAE physical-type identifier.
        legacy_unit : {"Msun", "kpc", "Msun/pc3", "dimensionless"}
            Unit attached implicitly by the legacy return contract.

        Returns
        -------
        numpy.ndarray
            Plain floating values in ITAMAE canonical units.
        """

        try:
            scale = _CANONICAL_SCALE[legacy_unit]
        except KeyError as error:
            raise ValueError(f"Unsupported legacy unit: {legacy_unit}") from error
        canonical = np.asarray(value, dtype=float) * scale

        if getattr(self.itamae_units, "identifier", "").startswith("astropy"):
            import astropy.units as u

            if physical_type == "dimensionless":
                quantity = canonical * u.dimensionless_unscaled
            else:
                quantity = canonical * u.Unit(_CANONICAL_ASTROPY_UNIT[physical_type])
            return self.itamae_units.to_internal(quantity, physical_type)
        return self.itamae_units.to_internal(canonical, physical_type)

    def rs_rhos_catalog_calc(self, *args: Any, **kwargs: Any) -> WeightedSubhaloCatalog:
        """Calculate a WDM catalog with exact NFW inversion and canonical units.

        The legacy calculation is executed unchanged except for compatibility
        aliases needed by modern NumPy/SciPy and the recognizable NFW inverse
        interpolation table. All temporary module changes are restored.
        """

        old_interp1d = _legacy.interp1d
        had_alen = hasattr(np, "alen")
        old_alen = getattr(np, "alen", None)
        had_simps = hasattr(_legacy.integrate, "simps")
        old_simps = getattr(_legacy.integrate, "simps", None)

        def migration_interp1d(x, y, *interp_args: Any, **interp_kwargs: Any):
            x_array = np.asarray(x)
            y_array = np.asarray(y)
            is_nfw_inverse = (
                y_array.ndim == 1
                and y_array.size == 1000
                and np.isclose(y_array[0], 0.0)
                and np.isclose(y_array[-1], 100.0)
                and x_array.shape == y_array.shape
                and np.allclose(x_array, self.fc(y_array), rtol=2.0e-13, atol=2.0e-15)
            )
            if is_nfw_inverse:
                return invert_nfw_mass_function
            return old_interp1d(x, y, *interp_args, **interp_kwargs)

        _legacy.interp1d = migration_interp1d
        if not had_alen:
            np.alen = len
        if not had_simps:
            _legacy.integrate.simps = _legacy.integrate.simpson
        try:
            result = super().rs_rhos_calc(*args, **kwargs)
        finally:
            _legacy.interp1d = old_interp1d
            if had_alen:
                np.alen = old_alen
            else:
                delattr(np, "alen")
            if had_simps:
                _legacy.integrate.simps = old_simps
            else:
                delattr(_legacy.integrate, "simps")

        return self.catalog_from_legacy(result)

    def catalog_from_legacy(self, result) -> WeightedSubhaloCatalog:
        """Convert the ten-element legacy WDM tuple to an ITAMAE catalog.

        Parameters
        ----------
        result : tuple
            Output of ``subhalos.rs_rhos_calc``. Its implicit units are
            ``Msun``, ``kpc``, and ``Msun/pc^3`` as documented by the legacy
            implementation.

        Returns
        -------
        itamae.types.WeightedSubhaloCatalog
            Catalog in canonical ITAMAE units with explicit survival weight.
        """

        if len(result) != 10:
            raise ValueError(f"Expected 10 legacy outputs, received {len(result)}.")
        survive = np.asarray(result[9], dtype=bool)
        columns = {
            "m200_acc": self._to_canonical(result[0], "mass", "Msun"),
            "z_acc": self._to_canonical(result[1], "dimensionless", "dimensionless"),
            "r_s_acc": self._to_canonical(result[2], "length", "kpc"),
            "rho_s_acc": self._to_canonical(result[3], "density", "Msun/pc3"),
            "m_bound": self._to_canonical(result[4], "mass", "Msun"),
            "r_s": self._to_canonical(result[5], "length", "kpc"),
            "rho_s": self._to_canonical(result[6], "density", "Msun/pc3"),
            "c_t": self._to_canonical(result[7], "dimensionless", "dimensionless"),
            "survive": survive,
        }
        legacy_weight = np.asarray(result[8], dtype=float)
        backend_config = BackendConfig(self.itamae_cosmology, self.itamae_units)
        return WeightedSubhaloCatalog(
            columns=columns,
            weights={
                "weight_base": legacy_weight,
                "weight_survival": survive.astype(float),
            },
            metadata={
                "schema_version": CATALOG_SCHEMA_VERSION,
                "model_identifier": (
                    f"sashimi-w:wdm:m_wdm_keV={self.mass_wdm:g}:itamae-migration:v1"
                ),
                "backend_identifier": backend_config.identifier,
                "source_identifier": "sashimi-w:itamae-migration",
                "mass_wdm_keV": float(self.mass_wdm),
                "unit_backend": self.itamae_units.identifier,
                "canonical_units": {
                    "mass": "Msun",
                    "length": "Mpc",
                    "density": "Msun/Mpc^3",
                },
                "legacy_units": {
                    "mass": "Msun",
                    "length": "kpc",
                    "density": "Msun/pc^3",
                },
                "legacy_weight_excludes_survival": True,
            },
        )


__all__ = ["ItamaeSubhalos"]
