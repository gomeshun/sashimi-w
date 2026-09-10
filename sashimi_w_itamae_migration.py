"""Incremental ITAMAE migration layer for SASHIMI-W.

SASHIMI-W differs materially from the C/SI/F implementations: it uses CGS
module constants, WMAP7 parameters loaded from a tabulated power spectrum, a
sharp-k variance calculation, and a distinct concentration prescription. This
adapter therefore keeps all WDM population physics in ``sashimi_w.py`` while
migrating the growth-factor boundary, NFW inversion, unit conversion, and
weighted-catalog representation.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from itamae.backends import BackendConfig
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import invert_nfw_mass_function
from itamae.provenance import build_migration_metadata
from itamae.protocols import CosmologyBackend
from itamae.types import WeightedSubhaloCatalog
from itamae.units import NativeUnits
import sashimi_w as _legacy
from sashimi_w import OmegaM, WDM_TRANSFER_NU, h, subhalos


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

_PHYSICS_MODES = {"consistent", "legacy"}
PUBLISHED_Q5 = "published-q5"
STANDARD_T2_Q10 = "standard-t2-q10"

# Migration decision record
# -------------------------
# The standard q10 formula is not coupled to ``physics_mode="consistent"`` and
# is not an implicit default.  Replacing the published q5 spectrum changes the
# EPS derivative and compact full-catalog abundance by orders of magnitude even
# after matching a half-mode scale.  Moreover, Ono et al. (2025) write the q10
# spectrum but do not identify the exact SASHIMI-W source commit used for their
# comparison, while the public upstream implementation remains q5.  Requiring
# this independent argument preserves published provenance and lets q10 be
# validated against simulation curves and likelihood results before any future
# model release selects it by default.
_WDM_POWER_CONVENTIONS = {
    PUBLISHED_Q5: {
        "q": 5.0,
        "formula_role": "q5-expression-used-as-power-ratio",
        "half_mode_definition": "P_WDM/P_CDM=0.5",
        "half_mode_power_ratio": 0.5,
    },
    STANDARD_T2_Q10: {
        "q": 10.0,
        "formula_role": "viel-transfer-amplitude-squared",
        "half_mode_definition": "T_WDM/T_CDM=0.5;P_WDM/P_CDM=0.25",
        "half_mode_power_ratio": 0.25,
    },
}
_CONSISTENT_OMEGA_L = 1.0 - float(OmegaM)
_LEGACY_OMEGA_L = float(_legacy.OmegaL)
_LEGACY_CATALOG_SIGNATURE = inspect.signature(subhalos.rs_rhos_calc)


class ItamaeSubhalos(subhalos):
    """SASHIMI-W model with explicit ITAMAE cosmology and unit boundaries.

    Parameters
    ----------
    mass_wdm : float, optional
        Warm-dark-matter particle mass in keV, following the legacy API.
    wdm_power_convention : {"published-q5", "standard-t2-q10"}
        Explicit power-suppression convention. ``"published-q5"`` reproduces
        arXiv:2111.13137 and the public implementation, which inserted the
        Viel q=5 expression directly as a power ratio. ``"standard-t2-q10"``
        treats that expression as a transfer amplitude and squares it.
    cosmology_backend : object, optional
        ITAMAE-compatible cosmology backend. The default uses the WMAP7 matter
        density and reduced Hubble parameter read by ``sashimi_w``.
    unit_backend : object, optional
        ITAMAE-compatible unit backend. Native canonical floating-point units
        are used by default.
    physics_mode : {"consistent", "legacy"}, optional
        ``"consistent"`` uses a flat WMAP7 background, normalizes ``D(0)=1``,
        and scales ``dS/dM`` with ``D(z)^2``. ``"legacy"`` reproduces the
        historical mixed-density and single-growth-factor conventions.

    Notes
    -----
    The legacy ``linear_growth_factor`` method has an unusual two-element input
    convention: a two-element redshift array represents ``(z1, z2)`` and returns
    ``D(z2) / D(z1)``, whereas arrays of any other size are normalized at zero.
    This adapter preserves that behavior.

    ``wdm_power_convention`` is deliberately independent of ``physics_mode``:
    expansion/growth corrections cannot silently opt callers into a different
    primordial power spectrum. Both conventions feed the sharp-k EPS variance
    and the top-hat Ludlow concentration calculation together. The q10 choice
    is physically standard but remains an explicitly selected validation path;
    this migration does not claim that existing q5-calibrated constraints can
    be converted or reinterpreted without rerunning the complete analysis.

    Legacy catalog output mixes solar masses, kiloparsecs, and solar masses per
    cubic parsec. ``rs_rhos_catalog_calc`` converts them to ITAMAE canonical
    units: solar masses, megaparsecs, and solar masses per cubic megaparsec.
    """

    def __init__(
        self,
        mass_wdm: float = 1.5,
        *,
        wdm_power_convention: str,
        cosmology_backend: Any | None = None,
        unit_backend: Any | None = None,
        physics_mode: str = "consistent",
    ) -> None:
        if physics_mode not in _PHYSICS_MODES:
            raise ValueError(f"physics_mode must be one of {sorted(_PHYSICS_MODES)}.")
        try:
            particle_mass = float(mass_wdm)
        except (TypeError, ValueError) as error:
            raise ValueError("mass_wdm must be finite and positive.") from error
        if not np.isfinite(particle_mass) or particle_mass <= 0.0:
            raise ValueError("mass_wdm must be finite and positive.")
        try:
            convention = _WDM_POWER_CONVENTIONS[wdm_power_convention]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"wdm_power_convention must be one of {sorted(_WDM_POWER_CONVENTIONS)}."
            ) from error
        backend = cosmology_backend or NativeFlatLCDM(omega_m0=float(OmegaM), h=float(h))
        self._validate_cosmology(backend)
        self.physics_mode = physics_mode
        self.wdm_power_convention = wdm_power_convention
        self.wdm_power_q = float(convention["q"])
        self.wdm_power_formula_role = str(convention["formula_role"])
        self.half_mode_definition = str(convention["half_mode_definition"])
        self.half_mode_power_ratio = float(convention["half_mode_power_ratio"])
        self.itamae_cosmology = backend
        self.itamae_units = unit_backend or NativeUnits()
        super().__init__(mass_wdm=particle_mass)
        self._consistent_variance_model = None

    def _wdm_power_ratio(self, k: Any, *, alpha: Any | None = None):
        """Return the explicitly selected WDM-to-CDM power ratio.

        Viel et al. define ``[1 + (alpha*k)^(2*nu)]^(-5/nu)`` as the
        transfer amplitude ``T=sqrt(P_WDM/P_CDM)``. Published SASHIMI-W used
        it directly as a power ratio (q=5); the standard convention squares
        it (q=10). This single hook is consumed by both the inherited sharp-k
        table and its top-hat concentration calculation, preventing a mixed
        q5/q10 model.
        """
        selected_alpha = self.a if alpha is None else alpha
        wavenumber = np.asarray(k, dtype=float)
        return (1.0 + (selected_alpha * wavenumber) ** (2.0 * WDM_TRANSFER_NU)) ** (
            -self.wdm_power_q / WDM_TRANSFER_NU
        )

    def half_mode_wavenumber(self) -> float:
        """Return the convention's documented half-mode scale in ``h/Mpc``.

        Both supported conventions use the same numerical factor but assign
        it different physical roles. Published q5 calls the q5 power ratio
        one half. Standard q10 follows the later amplitude convention
        ``T=0.5``, for which the power ratio is one quarter. Metadata records
        the distinction so identically valued scales cannot be mixed silently.
        """
        factor = (2.0 ** (WDM_TRANSFER_NU / 5.0) - 1.0) ** (1.0 / (2.0 * WDM_TRANSFER_NU))
        return float(factor / self.a)

    @staticmethod
    def _validate_cosmology(backend: Any) -> None:
        """Require a complete backend matching the WMAP7 migration model."""
        if not isinstance(backend, CosmologyBackend):
            raise TypeError("cosmology_backend must implement the ITAMAE cosmology protocol.")
        omega_m0 = float(np.asarray(backend.omega_m(0.0)))
        backend_h = float(np.asarray(backend.H(0.0))) / 100.0
        if not np.isclose(omega_m0, float(OmegaM), rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"SASHIMI-W migration requires OmegaM={float(OmegaM)}; received {omega_m0}."
            )
        if not np.isclose(backend_h, float(h), rtol=0.0, atol=1.0e-12):
            raise ValueError(f"SASHIMI-W migration requires h={float(h)}; received {backend_h}.")

    @property
    def omega_lambda(self) -> float:
        """Return the selected legacy or flat-consistent dark-energy density."""
        return _CONSISTENT_OMEGA_L if self.physics_mode == "consistent" else _LEGACY_OMEGA_L

    def g(self, z: Any):
        """Return ``H(z)^2/H0^2`` under the selected physics convention."""
        if self.physics_mode == "legacy":
            return super().g(z)
        redshift = np.asarray(z, dtype=float)
        return float(OmegaM) * (1.0 + redshift) ** 3 + self.omega_lambda

    def Hz(self, z: Any):
        """Return the WMAP7 Hubble rate in legacy inverse-second units."""
        if self.physics_mode == "legacy":
            return super().Hz(z)
        ratio = np.asarray(self.itamae_cosmology.H(z), dtype=float) / float(
            np.asarray(self.itamae_cosmology.H(0.0))
        )
        return _legacy.H0 * ratio

    def Omegaz(self, parameters: Any, redshift: Any):
        """Return matter density while retaining the historical signature."""
        if self.physics_mode == "legacy":
            return super().Omegaz(parameters, redshift)
        return np.asarray(self.itamae_cosmology.omega_m(redshift), dtype=float)

    def growthD(self, z: Any):
        """Return the selected legacy or normalized-consistent growth factor."""
        if self.physics_mode == "legacy":
            return super().growthD(z)
        growth = np.asarray(self.itamae_cosmology.growth_factor(z), dtype=float)
        growth0 = float(np.asarray(self.itamae_cosmology.growth_factor(0.0), dtype=float))
        result = growth / growth0
        return float(result) if result.ndim == 0 else result

    def dOdz(self, z: Any):
        """Return the derivative of the flat-consistent dark-energy fraction."""
        if self.physics_mode == "legacy":
            return super().dOdz(z)
        redshift = np.asarray(z, dtype=float)
        denominator = self.omega_lambda + float(OmegaM) * (1.0 + redshift) ** 3
        return -self.omega_lambda * 3.0 * float(OmegaM) * (1.0 + redshift) ** 2 / denominator**2

    def dDdz(self, z: Any):
        """Differentiate the selected Carroll-Press-Turner growth factor."""
        if self.physics_mode == "legacy":
            return super().dDdz(z)
        redshift = np.asarray(z, dtype=float)
        omega_lz = self.omega_lambda / (self.omega_lambda + float(OmegaM) * (1.0 + redshift) ** 3)
        omega_mz = 1.0 - omega_lz
        phi_z = (
            omega_mz ** (4.0 / 7.0) - omega_lz + (1.0 + omega_mz / 2.0) * (1.0 + omega_lz / 70.0)
        )
        phi0 = (
            float(OmegaM) ** (4.0 / 7.0)
            - self.omega_lambda
            + (1.0 + float(OmegaM) / 2.0) * (1.0 + self.omega_lambda / 70.0)
        )
        domega_l_dz = self.dOdz(redshift)
        dphi_dz = domega_l_dz * (
            (-4.0 / 7.0) * omega_mz ** (-3.0 / 7.0)
            + (omega_mz - omega_lz) / 140.0
            + 1.0 / 70.0
            - 3.0 / 2.0
        )
        result = (phi0 / float(OmegaM)) * (
            -domega_l_dz / (phi_z * (1.0 + redshift))
            - omega_mz * (dphi_dz * (1.0 + redshift) + phi_z) / (phi_z**2 * (1.0 + redshift) ** 2)
        )
        return float(result) if result.ndim == 0 else result

    def _consistent_variance(self):
        """Lazily construct the canonical-unit analytic sharp-k variance."""
        if self._consistent_variance_model is None:
            # Local import avoids a module cycle: the variance module exposes
            # the public factory and imports this class for its default model.
            from sashimi_w_itamae_variance import make_integrated_variance_model

            self._consistent_variance_model = make_integrated_variance_model(self)
        return self._consistent_variance_model

    def sigmaMz(self, mass: Any, z: Any):
        """Evaluate the physical sharp-k integral in the model's mass domain.

        Sigma and dS/dM use the same continuous finite-domain integral; no
        coarse mass interpolation or monotonic projection is applied.
        """
        if self.physics_mode == "legacy":
            return super().sigmaMz(mass, z)
        mass_array, redshift = np.broadcast_arrays(
            np.asarray(mass, dtype=float),
            np.asarray(z, dtype=float),
        )
        if not np.all(np.isfinite(mass_array)) or np.any(mass_array <= 0.0):
            raise ValueError("Masses must be finite and positive.")
        physical_grid = np.asarray(self.filter_Mass, dtype=float) / float(h)
        if np.any(mass_array < physical_grid[0]) or np.any(mass_array > physical_grid[-1]):
            raise ValueError("Mass lies outside the canonical SASHIMI-W variance grid.")
        result = np.asarray(self._consistent_variance().sigma(mass_array, redshift), dtype=float)
        return float(result) if result.ndim == 0 else result

    def dsdm(self, mass: Any, z: Any):
        r"""Return ``dS/dM`` under the selected growth convention."""
        if self.physics_mode == "legacy":
            return super().dsdm(mass, z)
        result = np.asarray(
            self._consistent_variance().dvariance_dmass(mass, z),
            dtype=float,
        )
        return float(result) if result.ndim == 0 else result

    def _select_accretion_mass_grid(self, mass_by_redshift: Any, final_mass: Any):
        """Select exact legacy or redshift-resolved accretion masses.

        The historical routine passed the last redshift row to ``Na_calc`` for
        every redshift. Legacy mode retains that bug-for-bug behavior.
        Consistent mode uses the already calculated row for each accretion
        redshift, so the EPS factors and their mass derivative are evaluated at
        the matching virial mass.
        """
        if self.physics_mode == "legacy":
            return super()._select_accretion_mass_grid(
                mass_by_redshift,
                final_mass,
            )
        mass_grid = np.asarray(mass_by_redshift, dtype=float)
        if mass_grid.ndim != 2 or not np.all(np.isfinite(mass_grid)) or np.any(mass_grid <= 0.0):
            raise ValueError("Accretion mass grid must be finite, positive, and two-dimensional.")
        return mass_grid

    def conc200(self, mass_cgs: Any, z: Any):
        """Evaluate concentration with an explicit physical-Msun boundary.

        The inherited Ludlow implementation stores its variance grid in
        numerical ``Msun/h`` units but receives a CGS mass. Its historical
        ``M / Msolar / h`` conversion is retained in legacy mode. For the
        consistent path, multiplying the CGS argument by ``h**2`` before
        delegating makes that inherited expression evaluate
        ``h * M / Msolar``, the correct physical-Msun to ``Msun/h`` mapping.
        """
        if self.physics_mode == "legacy":
            return super().conc200(mass_cgs, z)
        return super().conc200(np.asarray(mass_cgs, dtype=float) * float(h) ** 2, z)

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
            return np.asarray(backend.growth_factor(z2)) / np.asarray(backend.growth_factor(z1))

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
        """Execute WDM-owned stages with canonical arrays and independent factors."""
        bound = _LEGACY_CATALOG_SIGNATURE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        parameters = dict(bound.arguments)
        self._validate_catalog_inputs(parameters)
        if self.physics_mode == "legacy":
            return self.catalog_from_legacy(super().rs_rhos_calc(*args, **kwargs))
        return self._execute_population(parameters).to_catalog(self._catalog_metadata())

    def rs_rhos_calc(self, *args: Any, **kwargs: Any):
        """Return the historical tuple format of the current population calculation."""
        if self.physics_mode == "legacy":
            return super().rs_rhos_calc(*args, **kwargs)
        catalog = self.rs_rhos_catalog_calc(*args, **kwargs)
        columns = catalog.columns
        return (
            columns["m200_acc"],
            columns["z_acc"],
            columns["r_s_acc"] * 1e3,
            columns["rho_s_acc"] / 1e18,
            columns["m_bound"],
            columns["r_s"] * 1e3,
            columns["rho_s"] / 1e18,
            columns["c_t"],
            catalog.weights["weight_base"] * catalog.weights["weight_concentration"],
            columns["survive"],
        )

    def _execute_population(self, parameters):
        from scipy.integrate import simpson
        from itamae.execution import PopulationComponents
        from sashimi_w_itamae_components import (
            WDMAccretionSlices,
            WDMCatalogColumns,
            WDMHostHistory,
            WDMInitialStructure,
            WDMProfileEvolution,
            WDMSurvival,
            WDMTidalMassLoss,
        )

        p = parameters
        zdist = np.arange(p["redshift"] + p["dz"], p["zmax"] + p["dz"], p["dz"])
        logmax = np.log10(0.1 * p["M0"]) if p["logmamax"] is None else p["logmamax"]
        ma200 = np.logspace(p["logmamin"], logmax, p["N_ma"])
        ma_by_redshift = np.array(
            [self.Mvir_from_M200(ma200 * _legacy.Msolar, za) / _legacy.Msolar for za in zdist]
        )
        accretion = self.Na_calc(
            ma_by_redshift,
            zdist,
            p["M0"],
            z0=0,
            N_herm=p["N_hermNa"],
            Nrand=1000,
            sigmafac=p["sigmafac"],
        )
        total = simpson(simpson(accretion, x=np.log(ma_by_redshift)), x=np.log(1 + zdist))
        population = accretion / (1.0 + zdist[:, None])
        # Preserve the documented global quadrature normalization before splitting.
        population = population / np.sum(population) * total
        slices = WDMAccretionSlices(
            self, ma200, ma_by_redshift, zdist, population, p["sigmalogc"], p["N_herm"]
        )
        history = WDMHostHistory(self, p["M0"], p["N_hermNa"], p["sigmafac"])
        components = PopulationComponents(
            initializer=WDMInitialStructure(self, p["N_herm"]),
            evolver=WDMProfileEvolution(
                self,
                WDMTidalMassLoss(self, history),
                p["redshift"],
                p["N_herm"],
                p["profile_change"],
            ),
            survival=WDMSurvival(),
            columns=WDMCatalogColumns(),
        )
        batches, contexts = zip(*(slices.build(i) for i in range(zdist.size)), strict=True)
        return components.execute(
            batches,
            contexts=contexts,
            diagnostics={"variant": "sashimi-w", "accretion_integral": total},
        )

    @staticmethod
    def _validate_catalog_inputs(parameters: dict[str, Any]) -> None:
        """Reject invalid physical and numerical catalog inputs early."""
        for name in (
            "M0",
            "redshift",
            "dz",
            "zmax",
            "sigmalogc",
            "logmamin",
            "sigmafac",
        ):
            value = np.asarray(parameters[name])
            if value.ndim != 0 or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite scalar.")
        if parameters["logmamax"] is not None:
            value = np.asarray(parameters["logmamax"])
            if value.ndim != 0 or not np.isfinite(float(value)):
                raise ValueError("logmamax must be None or a finite scalar.")
        if float(parameters["M0"]) <= 0.0:
            raise ValueError("M0 must be positive.")
        if float(parameters["redshift"]) < 0.0:
            raise ValueError("redshift must be nonnegative.")
        if float(parameters["dz"]) <= 0.0:
            raise ValueError("dz must be positive.")
        if float(parameters["zmax"]) <= float(parameters["redshift"]):
            raise ValueError("zmax must be greater than redshift.")
        if float(parameters["sigmalogc"]) < 0.0:
            raise ValueError("sigmalogc must be nonnegative.")
        for name in ("N_ma", "N_herm", "N_hermNa"):
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if int(parameters["N_ma"]) < 2:
            raise ValueError("N_ma must be at least two for mass integration.")
        if not isinstance(parameters["profile_change"], (bool, np.bool_)):
            raise TypeError("profile_change must be boolean.")
        effective_logmamax = (
            np.log10(0.1 * float(parameters["M0"]))
            if parameters["logmamax"] is None
            else float(parameters["logmamax"])
        )
        if float(parameters["logmamin"]) >= effective_logmamax:
            raise ValueError("logmamin must be smaller than logmamax.")

    def _invert_nfw_mass_fraction(self, enclosed_fraction: Any):
        """Select the exact or historical NFW inverse without global mutation."""
        if self.physics_mode == "legacy":
            return super()._invert_nfw_mass_fraction(enclosed_fraction)
        return invert_nfw_mass_function(enclosed_fraction)

    def catalog_from_legacy(
        self,
        result,
        *,
        weight_base: Any | None = None,
        weight_concentration: Any | None = None,
    ) -> WeightedSubhaloCatalog:
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
        if weight_base is None:
            weight_base = legacy_weight
        if weight_concentration is None:
            weight_concentration = np.ones(legacy_weight.shape, dtype=float)
        weight_base = np.asarray(weight_base, dtype=float)
        weight_concentration = np.asarray(weight_concentration, dtype=float)
        if (
            weight_base.shape != legacy_weight.shape
            or weight_concentration.shape != legacy_weight.shape
        ):
            raise ValueError("Factorized weights must match the legacy catalog shape.")
        if not np.allclose(
            weight_base * weight_concentration,
            legacy_weight,
            rtol=2.0e-15,
            atol=0.0,
        ):
            raise ValueError("Factorized weights do not reconstruct the legacy catalog weight.")
        negative_weight = legacy_weight < 0.0
        if np.any(negative_weight):
            count = int(np.count_nonzero(negative_weight))
            minimum = float(np.min(legacy_weight))
            if self.physics_mode == "legacy":
                # The old fixed-node sharp-k variance table is slightly
                # non-monotonic on its WDM plateau.  Exact legacy reproduction
                # must retain the resulting signed tuple weights, but an
                # ITAMAE weighted catalog represents nonnegative effective
                # counts.  Never clip or renormalize here: either operation
                # would silently create a third, scientifically unreviewed
                # model between legacy and consistent.
                raise ValueError(
                    "Legacy SASHIMI-W produced signed population weights "
                    f"({count} negative entries; minimum={minimum:.6e}). "
                    "Use rs_rhos_calc() for exact legacy tuple reproduction, "
                    "or physics_mode='consistent' for the corrected monotonic "
                    "sharp-k derivative. Weights were not clipped or renormalized."
                )
            raise ValueError(
                "Consistent SASHIMI-W produced negative population weights "
                f"({count} entries; minimum={minimum:.6e}); this violates the "
                "corrected catalog contract."
            )
        return WeightedSubhaloCatalog(
            columns=columns,
            weights={
                "weight_base": weight_base,
                "weight_concentration": weight_concentration,
                "weight_survival": survive.astype(float),
            },
            metadata=self._catalog_metadata(),
        )

    def _catalog_metadata(self):
        backend_config = BackendConfig(self.itamae_cosmology, self.itamae_units)
        backend_identifier = (
            backend_config.identifier
            if self.physics_mode == "consistent"
            else (
                "legacy-wmap7-module:"
                f"OmegaM={float(OmegaM):.17g};"
                f"OmegaL={_LEGACY_OMEGA_L:.17g};h={float(h):.17g}"
            )
        )
        model_revision = "v6" if self.physics_mode == "consistent" else "v4"
        return build_migration_metadata(
            variant="sashimi-w",
            distribution_name="sashimi-w",
            module_file=__file__,
            model_identifier=(
                f"sashimi-w:wdm:m_wdm_keV={self.mass_wdm:g}:"
                f"physics={self.physics_mode}:"
                f"power={self.wdm_power_convention}:"
                f"itamae-migration:{model_revision}"
            ),
            backend_identifier=backend_identifier,
            source_identifier="sashimi-w:itamae-migration",
            physics_mode=self.physics_mode,
            variance_identifier=(
                self._consistent_variance().identifier
                if self.physics_mode == "consistent"
                else f"sashimi-w:sharp-k:{self.wdm_power_convention}:physics={self.physics_mode}:v1"
            ),
            power_identifier=f"sashimi-w:{self.wdm_power_convention}:v1",
            solver_identifier="sashimi-w:tidal-stripping:nfw:v1",
            extra={
                "mass_wdm_keV": float(self.mass_wdm),
                "wdm_power_convention": self.wdm_power_convention,
                "wdm_power_formula_role": self.wdm_power_formula_role,
                "wdm_power_q": self.wdm_power_q,
                "wdm_transfer_nu": float(WDM_TRANSFER_NU),
                "half_mode_definition": self.half_mode_definition,
                "half_mode_power_ratio": self.half_mode_power_ratio,
                "half_mode_wavenumber_h_per_mpc": self.half_mode_wavenumber(),
                "omega_m0": float(OmegaM),
                "omega_lambda0": self.omega_lambda,
                "cosmology_parameters": {
                    "omega_m0": float(OmegaM),
                    "omega_lambda0": self.omega_lambda,
                    "h": float(h),
                },
                "growth_normalized_at_z0": self.physics_mode == "consistent",
                "variance_growth_power": (2 if self.physics_mode == "consistent" else 1),
                "variance_mass_unit": (
                    "Msun"
                    if self.physics_mode == "consistent"
                    else "legacy-raw-Msun-values-on-Msun/h-grid"
                ),
                "variance_power_units": (
                    {"wavenumber": "1/Mpc", "power": "Mpc^3", "density": "Msun/Mpc^3"}
                    if self.physics_mode == "consistent"
                    else {
                        "wavenumber": "h/Mpc",
                        "power": "(Mpc/h)^3",
                        "density": "(Msun/h)/(Mpc/h)^3",
                    }
                ),
                "physical_to_filter_mass": (
                    "M_filter[Msun/h] = h * M_physical[Msun]"
                    if self.physics_mode == "consistent"
                    else "legacy-unconverted-variance-and-concentration-divides-by-h"
                ),
                "accretion_mass_redshift_mapping": (
                    "per-redshift-virial-mass-grid"
                    if self.physics_mode == "consistent"
                    else "legacy-final-redshift-grid-reused"
                ),
                "population_weight_contract": (
                    "nonnegative-corrected-counts"
                    if self.physics_mode == "consistent"
                    else "exact-signed-tuple;structured-catalog-requires-nonnegative-grid"
                ),
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
                "legacy_mode_known_inconsistencies": (
                    [
                        "OmegaM+OmegaL!=1",
                        "D(0)!=1",
                        "dS/dM scales as D instead of D^2",
                        "physical Msun values are passed directly to the Msun/h variance grid",
                        "conc200 divides physical Msun by h instead of multiplying by h",
                        "all accretion redshifts reuse the final redshift virial-mass grid",
                        "fixed-node variance noise can create signed population weights",
                    ]
                    if self.physics_mode == "legacy"
                    else []
                ),
            },
        )


__all__ = ["ItamaeSubhalos", "PUBLISHED_Q5", "STANDARD_T2_Q10"]
