"""WDM-owned prescriptions executed at ITAMAE's canonical-unit boundary.

Batches/stage arrays use physical Msun, Mpc, Msun/Mpc^3. Only local WDM
kernel calls use the historical CGS convention. Redshift is the evolution
coordinate; no time or h rescaling is implicit in ITAMAE's controller.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from itamae.evolution import PicardDomain, PicardMassLossSolver, solve_evolution
from itamae.measure import build_accretion_batch
from itamae.numerics import gauss_hermite_lognormal
from ._physics import G, H0, Msolar, h, kpc, pOmega, rhocrit0

_MPC = 1000.0 * kpc
_DENSITY = Msolar / _MPC**3


@dataclass(frozen=True)
class WDMHostHistory:
    """Prepare the WDM virial host history and its selected scatter convention for one
    population run.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    mass : float or array_like
        Positive halo mass. Use solar masses with the canonical backend; a supplied variance
        prescription must use the same mass convention and factors of h.
    N_hermNa : int
        Positive Gauss-Hermite order for the host-history scatter in the EPS accretion rate.
    sigmafac : float
        Deterministic host-history offset in units of log-mass scatter when N_hermNa=1;
        low-redshift scatter is anchored at exactly z=1.

    Notes
    -----
    Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
    explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
    Redshift decreases from accretion to observation, so the positive dm/dz law removes
    mass.
    """

    model: Any
    mass: float
    N_hermNa: int
    sigmafac: float
    _parameters: tuple = field(init=False, repr=False)

    def __post_init__(self):
        # This component is created anew for each population calculation.
        # The selected host/background coefficients do not depend on ODE time.
        """Cache host-history coefficients that are independent of the ODE redshift.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        object.__setattr__(self, "_parameters", self.model._host_history_parameters(self.mass, 0))

    def mass_virial(self, z):
        """Evaluate the WDM virial host mass with the prescribed deterministic offset when
        selected.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Virial host mass in physical Msun, including the explicit N_hermNa=1 scatter rule.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        model = self.model
        mass = model._host_history_mass(self._parameters, z, 0)
        if self.N_hermNa == 1:
            logmass = np.log10(mass)
            scatter_high = 0.12 - 0.15 * np.log10(mass / self.mass)
            mass1 = model._host_history_mass(self._parameters, 1.0, 0.0)
            scatter1 = 0.12 - 0.15 * np.log10(mass1 / self.mass)
            scatter_low = scatter1 / np.log10(mass1 / self.mass) * np.log10(mass / self.mass)
            mass = 10 ** (logmass + self.sigmafac * np.where(z > 1.0, scatter_high, scatter_low))
            if self.sigmafac > 0.0:
                mass = np.where(mass < self.mass, mass, self.mass)
        return model.Mvir_from_M200(mass * Msolar, z) / Msolar


@dataclass(frozen=True)
class WDMTidalMassLoss:
    """Supply the WDM-owned redshift tidal-loss law on a prepared host history.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    history : WDMHostHistory
        Prepared physical virial host history, including the selected deterministic scatter
        rule when used.

    Notes
    -----
    Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
    explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
    Redshift decreases from accretion to observation, so the positive dm/dz law removes
    mass.
    """

    model: Any
    history: WDMHostHistory

    def rhs(self, z, mass):
        # Keep the supplied WDM fitting functions and arithmetic grouping.
        """Evaluate dm/dz from the WDM amplitude, mass-ratio exponent and virial dynamical time.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        mass : float or array_like
            Positive halo mass. Use solar masses with the canonical backend; a supplied variance
            prescription must use the same mass convention and factors of h.

        Returns
        -------
        array_like
            Positive mass derivative in Msun per unit redshift; integration toward lower z
            removes mass.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        host_mass = self.history.mass_virial(z)
        log_host_mass = np.log10(host_mass)
        log10a = (-0.0019 * log_host_mass + 0.045) * z + (0.0097 * log_host_mass - 0.313)
        amplitude = pow(10, log10a)
        exponent = (-5.55e-5 * log_host_mass + 1.43e-3) * z + (3.34e-4 * log_host_mass - 8.11e-3)
        omega = self.model.Omegaz(pOmega, z)
        dynamic_time = (
            1.628
            * pow(h, -1)
            * pow(self.model.Delc(omega - 1) / 178.0, -0.5)
            * pow(self.model.Hz(z) / H0, -1)
            * (86400 * 365 * 1e9)
        )
        return (
            amplitude
            * (mass / dynamic_time)
            * pow(mass / host_mass, exponent)
            * pow(self.model.Hz(z) * (1 + z), -1)
        )


class WDMTidalSolver:
    """WDM coefficient adapter; ITAMAE owns all Picard numerical operations.

    Masses are physical Msun. The historical scalar virial conversion is
    evaluated independently at each requested redshift, retaining its solve.

    Parameters
    ----------
    law : WDMTidalMassLoss
        WDM-owned redshift mass-loss equation and its reference host history.
    z_min : float
        Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
    z_max : float
        Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.

    Notes
    -----
    Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
    explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
    Redshift decreases from accretion to observation, so the positive dm/dz law removes
    mass.
    """

    def __init__(self, law, *, z_min, z_max):
        """Bind WDM-owned coefficients to the shared guarded endpoint integrator.

        Parameters
        ----------
        law : WDMTidalMassLoss
            WDM-owned redshift mass-loss equation and its reference host history.
        z_min : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        z_max : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        self.law = law
        self.backend = PicardMassLossSolver(
            self.rate,
            self.exponent,
            self.reference_mass,
            z_min=z_min,
            z_max=z_max,
            state_key=self.state_key,
            domain=PicardDomain((0.0, 7.0), (-24.0, 3.0)),
        )

    def state_key(self):
        """Identify the WDM transfer, background and prepared host state controlling table reuse.

        Returns
        -------
        tuple
            Process-local physical key including particle mass, convention, cosmology and
            host/scatter coefficients.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        model, history = self.law.model, self.law.history
        backend = model.itamae_cosmology
        return (
            float(model.mass_wdm),
            model.wdm_power_convention,
            id(backend),
            backend.identifier,
            h,
            H0,
            rhocrit0,
            history.mass,
            history.N_hermNa,
            history.sigmafac,
            tuple(float(np.asarray(x)) for x in history._parameters),
        )

    def reference_mass(self, z):
        """Evaluate scalar WDM virial conversions at each distinct requested epoch.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            Host virial mass in physical Msun with the redshift shape.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        redshift = np.asarray(z, dtype=float)
        # A 1d mass array with a 1d redshift array is not the old concentration
        # kernel's paired-input convention; keep the scalar physical calls.
        flat, inverse = np.unique(redshift.ravel(), return_inverse=True)
        values = np.asarray(
            [float(np.asarray(self.law.history.mass_virial(float(t))).item()) for t in flat]
        )
        result = values[inverse].reshape(redshift.shape)
        return float(result) if result.ndim == 0 else result

    def exponent(self, z):
        """Evaluate the fitted WDM mass-ratio exponent at each host epoch.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless zeta(M_host,z), retaining the WDM coefficient fit.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        loghost = np.log10(self.backend.Mzvir(z))
        return (-5.55e-5 * loghost + 1.43e-3) * z + (3.34e-4 * loghost - 8.11e-3)

    def rate(self, z):
        """Evaluate the fitted WDM log-mass loss coefficient per unit redshift.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Phi=A/[t_dyn*H*(1+z)], using WDM host/dynamical-time conventions.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        model = self.law.model
        loghost = np.log10(self.backend.Mzvir(z))
        amplitude = 10 ** ((-0.0019 * loghost + 0.045) * z + (0.0097 * loghost - 0.313))
        omega = model.Omegaz(pOmega, z)
        dynamic_time = (
            1.628
            * h**-1
            * (model.Delc(omega - 1) / 178.0) ** -0.5
            * (model.Hz(z) / H0) ** -1
            * (86400 * 365 * 1e9)
        )
        return amplitude / dynamic_time / (model.Hz(z) * (1 + z))

    def mass(self, ma, za, z, *, method="picard_table", **options):
        """Compute the endpoint mass by the selected shared Picard or direct log-mass solver.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        method : str
            picard_table (default), dop853 or odeint for population evolution; the Picard
            adapter itself accepts only the first two.
        **options : Mapping, optional
            Explicit numerical controls for the named solver or table. Unsupported or reserved
            keys are rejected by the receiving routine.

        Returns
        -------
        float or numpy.ndarray
            Bound mass in physical Msun; unsupported adapter methods are rejected.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        if method == "picard_table":
            return self.backend.endpoint(ma, za, z, **options)
        if method == "dop853":
            return self.backend.direct(ma, za, z, **options)
        raise ValueError(f"Invalid Picard adapter method: {method}")


@dataclass(frozen=True)
class WDMAccretionSlices:
    """Attach concentration quadrature to the per-redshift WDM virial accretion grid.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    ma200 : numpy.ndarray
        One-dimensional accretion M200 grid in physical Msun.
    ma_by_redshift : numpy.ndarray
        Virial accretion mass in physical Msun for each redshift/mass node.
    zdist : numpy.ndarray
        One-dimensional dimensionless accretion-redshift grid, paired with the population
        grid first axis.
    population : numpy.ndarray
        Nonnegative expected-count quadrature factors on the redshift/mass grid, before
        concentration weights.
    sigmalogc : float
        Nonnegative standard deviation of log10 concentration, in dex.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.

    Notes
    -----
    Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
    explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
    Redshift decreases from accretion to observation, so the positive dm/dz law removes
    mass.
    """

    model: Any
    ma200: np.ndarray
    ma_by_redshift: np.ndarray
    zdist: np.ndarray
    population: np.ndarray
    sigmalogc: float
    N_herm: int

    def build(self, index):
        """Construct one canonical-unit accretion batch from the WDM CGS structure boundary.

        Parameters
        ----------
        index : int
            Index of the selected accretion-redshift slice in zdist and population_2d.

        Returns
        -------
        tuple
            AccretionBatch plus context retaining virial radius in cm and virial mass in Msun.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        za = self.zdist[index]
        ma = self.ma_by_redshift[index]
        omega = self.model.Omegaz(pOmega, za)
        c200 = self.model.conc200(self.ma200 * Msolar, za)
        rvir = pow(
            3
            * ma
            * Msolar
            * pow(rhocrit0 * self.model.g(za) * self.model.Delc(omega - 1) * 4 * np.pi, -1),
            1.0 / 3.0,
        )
        r200 = pow(
            3 * self.ma200 * Msolar * pow(rhocrit0 * self.model.g(za) * 200 * 4 * np.pi, -1),
            1.0 / 3.0,
        )
        concentration, weight = gauss_hermite_lognormal(
            c200 * rvir / r200, self.sigmalogc, order=self.N_herm
        )
        batch = build_accretion_batch(
            self.ma200,
            za,
            concentration,
            self.population[index],
            weight,
            mvir_acc=ma,
            metadata={"model": "sashimi-w", "physical_units": "Msun,Mpc,Msun/Mpc^3"},
        )
        return batch, {"rvir_cgs": rvir, "ma": ma, "z_acc": za}


@dataclass(frozen=True)
class WDMInitialStructure:
    """Normalize each accreted WDM NFW profile at its virial mass and concentration.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.

    Notes
    -----
    Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
    explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
    Redshift decreases from accretion to observation, so the positive dm/dz law removes
    mass.
    """

    model: Any
    N_herm: int

    def initialize(self, batch, context):
        """Convert NFW scale radius/density from CGS to canonical stage arrays.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Flattened r_s_acc in Mpc and rho_s_acc in Msun/Mpc^3.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        concentration = batch.concentration_acc.reshape(self.N_herm, -1)
        radius_cgs = context["rvir_cgs"] / concentration
        density_cgs = (
            context["ma"] * Msolar / (4 * np.pi * radius_cgs**3 * self.model.fc(concentration))
        )
        return {
            "r_s_acc": (radius_cgs / _MPC).reshape(-1),
            "rho_s_acc": (density_cgs / _DENSITY).reshape(-1),
        }


@dataclass(frozen=True)
class WDMProfileEvolution:
    """Apply the selected tidal solver and optional NFW structural response at the observation
    epoch.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    law : WDMTidalMassLoss
        WDM-owned redshift mass-loss equation and its reference host history.
    redshift : float or array_like
        Dimensionless redshift. The cosmological background requires z > -1; population
        nodes use nonnegative redshifts.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.
    profile_change : bool
        Apply the variant tidal response of scale radius and density in addition to
        bound-mass loss.
    method : str
        picard_table (default), dop853 or odeint for population evolution; the Picard
        adapter itself accepts only the first two.
    solver_options : dict
        Explicit numerical keyword controls for the selected tidal method; stored in
        calculation provenance.
    solver : WDMTidalSolver, optional
        WDM coefficient adapter preserving the physical host history; the evolution stage
        can construct one when absent.

    Notes
    -----
    Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
    explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
    Redshift decreases from accretion to observation, so the positive dm/dz law removes
    mass.
    """

    model: Any
    law: WDMTidalMassLoss
    redshift: float
    N_herm: int
    profile_change: bool
    method: str = "picard_table"
    solver_options: dict = field(default_factory=dict)
    solver: Any = None

    def evolve(self, batch, initial, context):
        """Evolve bound mass, scale parameters and the mass-conserving NFW tidal radius.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Flattened m_bound (Msun), r_s (Mpc), rho_s (Msun/Mpc^3) and dimensionless c_t.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        ma = context["ma"]
        if self.method == "odeint":
            options = dict(self.solver_options)
            rtol, atol = options.pop("rtol", None), options.pop("atol", None)
            grid = np.linspace(context["z_acc"], self.redshift, 100)
            bound_mass = solve_evolution(
                self.law.rhs,
                ma,
                grid,
                method="odeint",
                rtol=rtol,
                atol=atol,
                odeint_options=options,
            )[-1]
        else:
            solver = self.solver or WDMTidalSolver(
                self.law, z_min=self.redshift, z_max=context["z_acc"]
            )
            bound_mass = solver.mass(
                ma, context["z_acc"], self.redshift, method=self.method, **self.solver_options
            )
        rs_acc = initial["r_s_acc"].reshape(self.N_herm, -1) * _MPC
        rhos_acc = initial["rho_s_acc"].reshape(self.N_herm, -1) * _DENSITY
        if self.profile_change:
            rmax_acc = rs_acc * 2.163
            vmax_acc = np.sqrt(rhos_acc * 4 * np.pi * G / 4.625) * rs_acc
            vmax = vmax_acc * (
                pow(2, 0.4) * pow(bound_mass / ma, 0.3) * pow(1 + bound_mass / ma, -0.4)
            )
            rmax = rmax_acc * (
                pow(2, -0.3) * pow(bound_mass / ma, 0.4) * pow(1 + bound_mass / ma, 0.3)
            )
            radius = rmax / 2.163
            density = (4.625 / (4 * np.pi * G)) * pow(vmax / radius, 2)
        else:
            radius, density = rs_acc, rhos_acc
        concentration = self.model._invert_nfw_mass_fraction(
            bound_mass * Msolar / (4 * np.pi * density * radius**3)
        )
        return {
            "m_bound": np.broadcast_to(bound_mass, radius.shape).reshape(-1),
            "r_s": (radius / _MPC).reshape(-1),
            "rho_s": (density / _DENSITY).reshape(-1),
            "c_t": concentration.reshape(-1),
        }


@dataclass(frozen=True)
class WDMSurvival:
    """Select surviving WDM nodes by the strict tidal-concentration threshold.

    Parameters
    ----------
    ct_threshold : float
        Dimensionless tidal truncation threshold r_t/r_s; the specified strict survival rule
        is applied once through its mask.

    Notes
    -----
    Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
    explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
    Redshift decreases from accretion to observation, so the positive dm/dz law removes
    mass.
    """

    ct_threshold: float = 0.77

    def select(self, batch, initial, evolved, context):
        """Apply c_t > ct_threshold without changing generation weights or physical columns.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        numpy.ndarray of bool
            Survival mask aligned with the original nodes.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        return evolved["c_t"] > self.ct_threshold


@dataclass(frozen=True)
class WDMCatalogColumns:
    """Name the canonical accretion/evolved WDM fields and their survival mask.

    Notes
    -----
    Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
    explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
    Redshift decreases from accretion to observation, so the positive dm/dz law removes
    mass.
    """

    def build(self, batch, initial, evolved, survival_masks, context):
        """Preserve node alignment while exposing accretion and evolved physical fields.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.
        survival_masks : Mapping[str, numpy.ndarray]
            Named boolean survival decisions aligned with the physical state; kept separate from
            population and concentration weights.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Canonical-unit columns plus the default boolean survival mask.

        Notes
        -----
        Population stages use physical Msun, Mpc and Msun/Mpc^3. Local structure calls
        explicitly convert to CGS; no time or h conversion is implicit in the shared solver.
        Redshift decreases from accretion to observation, so the positive dm/dz law removes
        mass.
        """
        return {
            "m200_acc": batch.m200_acc,
            "z_acc": batch.z_acc,
            "r_s_acc": initial["r_s_acc"],
            "rho_s_acc": initial["rho_s_acc"],
            **evolved,
            "survive": survival_masks["default"],
        }
