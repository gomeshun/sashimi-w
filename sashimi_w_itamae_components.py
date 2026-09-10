"""WDM-owned prescriptions executed at ITAMAE's canonical-unit boundary.

Batches/stage arrays use physical Msun, Mpc, Msun/Mpc^3. Only local WDM
kernel calls use the historical CGS convention. Redshift is the evolution
coordinate; no time or h rescaling is implicit in ITAMAE's controller.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from itamae.evolution import solve_evolution
from itamae.measure import build_accretion_batch
from itamae.numerics import gauss_hermite_lognormal
from sashimi_w_physics import G, H0, Msolar, h, kpc, pOmega, rhocrit0

_MPC = 1000.0 * kpc
_DENSITY = Msolar / _MPC**3


@dataclass(frozen=True)
class WDMHostHistory:
    model: Any
    mass: float
    N_hermNa: int
    sigmafac: float

    def mass_virial(self, z):
        model = self.model
        mass = model.Mzzi(self.mass, z, 0)
        if self.N_hermNa == 1:
            logmass = np.log10(mass)
            scatter_high = 0.12 - 0.15 * np.log10(mass / self.mass)
            mass1 = model.Mzzi(self.mass, 1.0, 0.0)
            scatter1 = 0.12 - 0.15 * np.log10(mass1 / self.mass)
            scatter_low = scatter1 / np.log10(mass1 / self.mass) * np.log10(mass / self.mass)
            mass = 10 ** (logmass + self.sigmafac * np.where(z > 1.0, scatter_high, scatter_low))
            if self.sigmafac > 0.0:
                mass = np.where(mass < self.mass, mass, self.mass)
        return model.Mvir_from_M200(mass * Msolar, z) / Msolar


@dataclass(frozen=True)
class WDMTidalMassLoss:
    model: Any
    history: WDMHostHistory

    def rhs(self, z, mass):
        # Keep the supplied WDM fitting functions and arithmetic grouping.
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


@dataclass(frozen=True)
class WDMAccretionSlices:
    model: Any
    ma200: np.ndarray
    ma_by_redshift: np.ndarray
    zdist: np.ndarray
    population: np.ndarray
    sigmalogc: float
    N_herm: int

    def build(self, index):
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
    model: Any
    N_herm: int

    def initialize(self, batch, context):
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
    model: Any
    law: WDMTidalMassLoss
    redshift: float
    N_herm: int
    profile_change: bool

    def evolve(self, batch, initial, context):
        ma = context["ma"]
        grid = np.linspace(context["z_acc"], self.redshift, 100)
        bound_mass = solve_evolution(self.law.rhs, ma, grid, method="odeint")[-1]
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
    ct_threshold: float = 0.77

    def select(self, batch, initial, evolved, context):
        return evolved["c_t"] > self.ct_threshold


@dataclass(frozen=True)
class WDMCatalogColumns:
    def build(self, batch, initial, evolved, survival_masks, context):
        return {
            "m200_acc": batch.m200_acc,
            "z_acc": batch.z_acc,
            "r_s_acc": initial["r_s_acc"],
            "rho_s_acc": initial["rho_s_acc"],
            **evolved,
            "survive": survival_masks["default"],
        }
