"""Standard ITAMAE-backed WDM calculation and named catalog API."""

from __future__ import annotations
from typing import Any
import numpy as np
from itamae.backends import BackendConfig
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import invert_nfw_mass_function
from itamae.provenance import build_calculation_metadata
from itamae.protocols import CosmologyBackend
from itamae.types import WeightedSubhaloCatalog
from itamae.units import NativeUnits
import sashimi_w_physics as _physics
from sashimi_w_physics import OmegaM, WDM_TRANSFER_NU, h, WDMPhysics

_CANONICAL_SCALE = {"Msun": 1.0, "kpc": 0.001, "Msun/pc3": 1e18, "dimensionless": 1.0}
_CANONICAL_ASTROPY_UNIT = {"mass": "Msun", "length": "Mpc", "density": "Msun / Mpc3"}
PUBLISHED_Q5 = "published-q5"
STANDARD_T2_Q10 = "standard-t2-q10"
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


class Subhalos(WDMPhysics):
    """WDM model with explicit q5/q10 power choice and a fixed calculation specification.

    Canonical catalogs use Msun, Mpc and Msun/Mpc^3. Tuple conversion uses the
    historical Msun, kpc and Msun/pc^3 format. No legacy calculation is provided."""

    def __init__(
        self,
        mass_wdm: float = 1.5,
        *,
        wdm_power_convention: str,
        cosmology_backend: Any | None = None,
        unit_backend: Any | None = None,
    ) -> None:
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
        self.wdm_power_convention = wdm_power_convention
        self.wdm_power_q = float(convention["q"])
        self.wdm_power_formula_role = str(convention["formula_role"])
        self.half_mode_definition = str(convention["half_mode_definition"])
        self.half_mode_power_ratio = float(convention["half_mode_power_ratio"])
        self.itamae_cosmology = backend
        self.itamae_units = unit_backend or NativeUnits()
        super().__init__(mass_wdm=particle_mass)
        self._variance_model = None

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
        if not np.isclose(omega_m0, float(OmegaM), rtol=0.0, atol=1e-12):
            raise ValueError(
                f"SASHIMI-W migration requires OmegaM={float(OmegaM)}; received {omega_m0}."
            )
        if not np.isclose(backend_h, float(h), rtol=0.0, atol=1e-12):
            raise ValueError(f"SASHIMI-W migration requires h={float(h)}; received {backend_h}.")

    @property
    def omega_lambda(self) -> float:
        """Flat WMAP7 dark-energy density fraction."""
        return _CONSISTENT_OMEGA_L

    def g(self, z: Any):
        """Return ``H(z)^2/H0^2`` under the selected physics convention."""
        redshift = np.asarray(z, dtype=float)
        return float(OmegaM) * (1.0 + redshift) ** 3 + self.omega_lambda

    def Hz(self, z: Any):
        """Return the WMAP7 Hubble rate in historical inverse-second units."""
        ratio = np.asarray(self.itamae_cosmology.H(z), dtype=float) / float(
            np.asarray(self.itamae_cosmology.H(0.0))
        )
        return _physics.H0 * ratio

    def Omegaz(self, parameters: Any, redshift: Any):
        """Return matter density while retaining the historical signature."""
        return np.asarray(self.itamae_cosmology.omega_m(redshift), dtype=float)

    def growthD(self, z: Any):
        """Return the normalized WMAP7 growth factor."""
        growth = np.asarray(self.itamae_cosmology.growth_factor(z), dtype=float)
        growth0 = float(np.asarray(self.itamae_cosmology.growth_factor(0.0), dtype=float))
        result = growth / growth0
        return float(result) if result.ndim == 0 else result

    def dOdz(self, z: Any):
        """Return the derivative of the flat WMAP7 dark-energy fraction."""
        redshift = np.asarray(z, dtype=float)
        denominator = self.omega_lambda + float(OmegaM) * (1.0 + redshift) ** 3
        return -self.omega_lambda * 3.0 * float(OmegaM) * (1.0 + redshift) ** 2 / denominator**2

    def dDdz(self, z: Any):
        """Differentiate the selected Carroll-Press-Turner growth factor."""
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
            -4.0 / 7.0 * omega_mz ** (-3.0 / 7.0)
            + (omega_mz - omega_lz) / 140.0
            + 1.0 / 70.0
            - 3.0 / 2.0
        )
        result = (
            phi0
            / float(OmegaM)
            * (
                -domega_l_dz / (phi_z * (1.0 + redshift))
                - omega_mz
                * (dphi_dz * (1.0 + redshift) + phi_z)
                / (phi_z**2 * (1.0 + redshift) ** 2)
            )
        )
        return float(result) if result.ndim == 0 else result

    def variance_model(self):
        """Lazily construct the canonical-unit analytic sharp-k variance."""
        if self._variance_model is None:
            from sashimi_w_itamae_variance import make_integrated_variance_model

            self._variance_model = make_integrated_variance_model(self)
        return self._variance_model

    def sigmaMz(self, mass: Any, z: Any):
        """Evaluate the physical sharp-k integral in the model's mass domain.

        Sigma and dS/dM use the same continuous finite-domain integral; no
        coarse mass interpolation or monotonic projection is applied.
        """
        mass_array, redshift = np.broadcast_arrays(
            np.asarray(mass, dtype=float), np.asarray(z, dtype=float)
        )
        if not np.all(np.isfinite(mass_array)) or np.any(mass_array <= 0.0):
            raise ValueError("Masses must be finite and positive.")
        physical_grid = np.asarray(self.filter_Mass, dtype=float) / float(h)
        if np.any(mass_array < physical_grid[0]) or np.any(mass_array > physical_grid[-1]):
            raise ValueError("Mass lies outside the canonical SASHIMI-W variance grid.")
        result = np.asarray(self.variance_model().sigma(mass_array, redshift), dtype=float)
        return float(result) if result.ndim == 0 else result

    def dsdm(self, mass: Any, z: Any):
        """Return ``dS/dM`` under the selected growth convention."""
        result = np.asarray(self.variance_model().dvariance_dmass(mass, z), dtype=float)
        return float(result) if result.ndim == 0 else result

    def conc200(self, mass_cgs: Any, z: Any):
        """Ludlow concentration for CGS physical mass.

        The kernel uses M/Msolar/h; multiplying by h squared before that
        boundary implements M_filter[Msun/h] = h M_physical[Msun]."""
        return super().conc200(np.asarray(mass_cgs, dtype=float) * float(h) ** 2, z)

    def linear_growth_factor(self, Omega_m0: float, Omega_l0: float, z: Any):
        """Return the established interval-aware linear growth factor using ITAMAE.

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
        if not np.isclose(Omega_m0 + Omega_l0, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("The initial ITAMAE WDM adapter supports flat cosmologies only.")
        backend = self.itamae_cosmology
        backend_omega_m = getattr(backend, "omega_m0", Omega_m0)
        if not np.isclose(backend_omega_m, Omega_m0, rtol=0.0, atol=1e-12):
            backend = NativeFlatLCDM(omega_m0=float(Omega_m0), h=float(h))
        redshift = np.asarray(z, dtype=float)
        flattened = np.atleast_1d(redshift)
        if flattened.size == 2:
            z1, z2 = (flattened[0], flattened[1])
            return np.asarray(backend.growth_factor(z2)) / np.asarray(backend.growth_factor(z1))
        result = np.asarray(backend.growth_factor(redshift)) / np.asarray(
            backend.growth_factor(0.0)
        )
        return float(result) if result.ndim == 0 else result

    def _to_canonical(self, value, physical_type: str, tuple_unit: str) -> np.ndarray:
        """Convert a tuple array to the active ITAMAE unit backend.

        Parameters
        ----------
        value
            Legacy scalar or array.
        physical_type : str
            ITAMAE physical-type identifier.
        tuple_unit : {"Msun", "kpc", "Msun/pc3", "dimensionless"}
            Unit attached implicitly by the tuple return contract.

        Returns
        -------
        numpy.ndarray
            Plain floating values in ITAMAE canonical units.
        """
        try:
            scale = _CANONICAL_SCALE[tuple_unit]
        except KeyError as error:
            raise ValueError(f"Unsupported tuple unit: {tuple_unit}") from error
        canonical = np.asarray(value, dtype=float) * scale
        if getattr(self.itamae_units, "identifier", "").startswith("astropy"):
            import astropy.units as u

            if physical_type == "dimensionless":
                quantity = canonical * u.dimensionless_unscaled
            else:
                quantity = canonical * u.Unit(_CANONICAL_ASTROPY_UNIT[physical_type])
            return self.itamae_units.to_internal(quantity, physical_type)
        return self.itamae_units.to_internal(canonical, physical_type)

    def rs_rhos_catalog_calc(
        self,
        M0,
        redshift=0.0,
        dz=0.1,
        zmax=7.0,
        N_ma=100,
        sigmalogc=0.128,
        N_herm=5,
        logmamin=1,
        logmamax=None,
        sigmafac=0,
        N_hermNa=200,
        profile_change=True,
    ) -> WeightedSubhaloCatalog:
        """Calculate one named physical WDM catalog with explicit numerical inputs."""
        parameters = dict(locals())
        parameters.pop("self")
        self._validate_catalog_inputs(parameters)
        return self._execute_population(parameters).to_catalog(self._catalog_metadata(parameters))

    def rs_rhos_calc(self, *args: Any, **kwargs: Any):
        """Return the historical tuple format of the current population calculation."""
        catalog = self.rs_rhos_catalog_calc(*args, **kwargs)
        columns = catalog.columns
        return (
            columns["m200_acc"],
            columns["z_acc"],
            columns["r_s_acc"] * 1000.0,
            columns["rho_s_acc"] / 1e18,
            columns["m_bound"],
            columns["r_s"] * 1000.0,
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
            [self.Mvir_from_M200(ma200 * _physics.Msolar, za) / _physics.Msolar for za in zdist]
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
        for name in ("M0", "redshift", "dz", "zmax", "sigmalogc", "logmamin", "sigmafac"):
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
        """Exact positive NFW mass inverse from ITAMAE."""
        return invert_nfw_mass_function(enclosed_fraction)

    def catalog_from_tuple(
        self, result, *, weight_base: Any | None = None, weight_concentration: Any | None = None
    ) -> WeightedSubhaloCatalog:
        """Convert a ten-element tuple format; does not execute an old model."""
        if len(result) != 10:
            raise ValueError(f"Expected 10 tuple outputs, received {len(result)}.")
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
        tuple_weight = np.asarray(result[8], dtype=float)
        if weight_base is None:
            weight_base = tuple_weight
        if weight_concentration is None:
            weight_concentration = np.ones(tuple_weight.shape, dtype=float)
        weight_base = np.asarray(weight_base, dtype=float)
        weight_concentration = np.asarray(weight_concentration, dtype=float)
        if (
            weight_base.shape != tuple_weight.shape
            or weight_concentration.shape != tuple_weight.shape
        ):
            raise ValueError("Factorized weights must match the tuple catalog shape.")
        if not np.allclose(weight_base * weight_concentration, tuple_weight, rtol=2e-15, atol=0.0):
            raise ValueError("Factorized weights do not reconstruct the tuple catalog weight.")
        negative_weight = tuple_weight < 0.0
        if np.any(negative_weight):
            count = int(np.count_nonzero(negative_weight))
            minimum = float(np.min(tuple_weight))
            raise ValueError(
                f"Consistent SASHIMI-W produced negative population weights ({count} entries; minimum={minimum:.6e}); this violates the corrected catalog contract."
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

    def _catalog_metadata(self, parameters=None):
        backend = BackendConfig(self.itamae_cosmology, self.itamae_units)
        variance = self.variance_model()
        return build_calculation_metadata(
            variant="sashimi-w",
            distribution_name="sashimi-w",
            module_file=__file__,
            calculation_specification="sashimi-w:wdm:2026-09-10:v1",
            model_identifier=f"sashimi-w:wdm:m_wdm_keV={self.mass_wdm:g}:power={self.wdm_power_convention}:v1",
            backend_identifier=backend.identifier,
            source_identifier="sashimi-w:itamae-migration",
            variance_identifier=variance.identifier,
            power_identifier=variance.power.identifier,
            solver_identifier="itamae:odeint:scipy-default-tolerances:100-output-points",
            extra={
                "mass_wdm_keV": float(self.mass_wdm),
                "wdm_power_convention": self.wdm_power_convention,
                "wdm_power_formula_role": self.wdm_power_formula_role,
                "wdm_power_q": self.wdm_power_q,
                "wdm_transfer_nu": float(WDM_TRANSFER_NU),
                "half_mode_definition": self.half_mode_definition,
                "half_mode_power_ratio": self.half_mode_power_ratio,
                "half_mode_wavenumber_h_per_mpc": self.half_mode_wavenumber(),
                "cosmology_parameters": {
                    "omega_m0": float(OmegaM),
                    "omega_lambda0": self.omega_lambda,
                    "h": float(h),
                },
                "growth_normalized_at_z0": True,
                "variance_growth_power": 2,
                "variance_mass_unit": "Msun",
                "variance_power_units": {
                    "wavenumber": "1/Mpc",
                    "power": "Mpc^3",
                    "density": "Msun/Mpc^3",
                },
                "physical_to_filter_mass": "M_filter[Msun/h] = h * M_physical[Msun]",
                "accretion_mass_redshift_mapping": "per-redshift-virial-mass-grid",
                "population_weight_contract": "nonnegative-corrected-counts",
                "unit_backend": self.itamae_units.identifier,
                "canonical_units": {"mass": "Msun", "length": "Mpc", "density": "Msun/Mpc^3"},
                "tuple_units": {"mass": "Msun", "length": "kpc", "density": "Msun/pc^3"},
                "tuple_weight_excludes_survival": True,
                "calculation_parameters": {} if parameters is None else parameters,
                "survival": "strict c_t > 0.77",
                "profile": "tidally-evolved NFW",
                "concentration": "Ludlow-2016:100-mass-nodes:500-tophat-nodes",
            },
        )


ItamaeSubhalos = Subhalos
__all__ = ["Subhalos", "ItamaeSubhalos", "PUBLISHED_Q5", "STANDARD_T2_Q10"]
