"""Immutable WDM settings with explicit coupled power and host-history choices."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._native_support import (
    boolean,
    bounds,
    choice,
    freeze,
    integer,
    merge,
    ode_options,
    plain,
    redshift_grid,
    scalar,
    settings_identifier,
)

_DEFAULTS: dict[str, dict[str, Any]] = {
    "dark_matter": {"mass_keV": 1.5, "power_convention": "standard-t2-q10"},
    "accretion": {
        "prescription": "yang2011-wdm",
        "mass_nodes": 100,
        "redshift_step": 0.1,
        "host_history_mode": "quadrature",
        "host_history_nodes": 200,
        "sigmafac": 0.0,
    },
    "concentration": {
        "prescription": "ludlow2016-q10",
        "scatter_dex": 0.128,
        "quadrature_nodes": 5,
    },
    "stripping": {
        "prescription": "wdm-tidal",
        "solver": "odeint",
        "solver_options": {},
        "profile_change": True,
    },
    "disruption": {"prescription": "truncation", "ct_threshold": 0.77},
}


def _resolve(previous=None, **overrides):
    values = merge(_DEFAULTS, previous, overrides)
    dm, a, c, s, d = (values[key] for key in _DEFAULTS)
    for group in ("accretion", "concentration", "stripping", "disruption"):
        choice(
            values[group]["prescription"],
            group + ".prescription",
            (_DEFAULTS[group]["prescription"],),
        )
    dm["mass_keV"] = scalar(dm["mass_keV"], "mass_keV", positive=True)
    choice(dm["power_convention"], "power_convention", ("standard-t2-q10",))
    for group, key, minimum in (
        (a, "mass_nodes", 2),
        (a, "host_history_nodes", 1),
        (c, "quadrature_nodes", 1),
    ):
        group[key] = integer(group[key], key, minimum)
    a["redshift_step"] = scalar(a["redshift_step"], "redshift_step", positive=True)
    a["sigmafac"] = scalar(a["sigmafac"], "sigmafac", minimum=-np.inf)
    mode = choice(a["host_history_mode"], "host_history_mode", ("quadrature", "deterministic"))
    if mode == "deterministic" and a["host_history_nodes"] != 1:
        raise ValueError(
            "deterministic host_history_mode requires host_history_nodes=1; set both explicitly."
        )
    if mode == "quadrature" and (a["host_history_nodes"] < 2 or a["sigmafac"] != 0.0):
        raise ValueError(
            "quadrature host_history_mode requires nodes>=2 and sigmafac=0; reset inherited values explicitly."
        )
    c["scatter_dex"] = scalar(c["scatter_dex"], "scatter_dex")
    d["ct_threshold"] = scalar(d["ct_threshold"], "ct_threshold")
    choice(s["solver"], "solver", ("odeint",))
    s["solver_options"] = ode_options(s["solver_options"], s["solver"])
    s["profile_change"] = boolean(s["profile_change"], "profile_change")
    return freeze(values)


@dataclass(frozen=True, slots=True)
class _Preparation:
    redshift_nodes: np.ndarray
    solver_options: Mapping[str, Any]
    ct_threshold: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class WDM:
    """One immutable thermal-WDM specification with the standard q10 coupling."""

    _settings: Mapping[str, Any] = field(default_factory=_resolve, init=False, repr=False)

    @property
    def resolved_settings(self):
        return self._settings

    def configure(
        self,
        *,
        dark_matter=None,
        accretion=None,
        concentration=None,
        stripping=None,
        disruption=None,
    ):
        result = WDM()
        object.__setattr__(
            result,
            "_settings",
            _resolve(
                self._settings,
                dark_matter=dark_matter,
                accretion=accretion,
                concentration=concentration,
                stripping=stripping,
                disruption=disruption,
            ),
        )
        return result

    def population(
        self,
        *,
        host_mass_msun,
        host_mass_definition="200c",
        host_mass_redshift=0.0,
        redshift=0.0,
        accretion_mass_range_msun=(10.0, None),
        accretion_mass_definition="200c",
        accretion_redshift_range=None,
    ):
        """Return the weighted WDM catalog; host input is M200c at z=0 only.

        The particle mass and q10 power law are shared by sharp-k variance and
        the top-hat concentration relation. Deterministic host history is an
        explicit physical branch, not merely a lower quadrature resolution.
        """
        choice(host_mass_definition, "host_mass_definition", ("200c",))
        choice(accretion_mass_definition, "accretion_mass_definition", ("200c",))
        mass = scalar(host_mass_msun, "host_mass_msun", positive=True)
        epoch = scalar(host_mass_redshift, "host_mass_redshift")
        if epoch != 0.0:
            raise ValueError(
                "WDM currently requires host_mass_redshift=0; target-epoch mass inversion is unsupported."
            )
        target = scalar(redshift, "redshift")
        lo, hi = bounds(accretion_mass_range_msun, "accretion_mass_range_msun", mass=True)
        hi = 0.1 * mass if hi is None else hi
        if hi <= lo:
            raise ValueError("Resolved accretion mass upper bound must exceed lower bound.")
        dm, a, c, s, d = (self._settings[key] for key in _DEFAULTS)
        nodes, zlo, zhi, policy = redshift_grid(
            target, accretion_redshift_range, a["redshift_step"], 7.0
        )
        deterministic = a["host_history_mode"] == "deterministic"
        # PR19 anchors the deterministic width to the physical z=1 host mass,
        # independently of whether this requested redshift grid samples z=1.
        anchor = 1.0 if deterministic else None
        from ._itamae_migration import Subhalos

        model = Subhalos(mass_wdm=dm["mass_keV"], wdm_power_convention=dm["power_convention"])
        parameters = dict(
            M0=mass,
            redshift=target,
            dz=a["redshift_step"],
            zmax=zhi,
            N_ma=a["mass_nodes"],
            sigmalogc=c["scatter_dex"],
            N_herm=c["quadrature_nodes"],
            logmamin=np.log10(lo),
            logmamax=np.log10(hi),
            sigmafac=a["sigmafac"],
            N_hermNa=a["host_history_nodes"],
            profile_change=s["profile_change"],
        )
        model._validate_catalog_inputs(parameters)
        metadata = {
            "native_api": "sashimi-w:immutable-specification:v1",
            "resolved_settings": plain(self._settings),
            "native_configuration_identifier": settings_identifier("sashimi-w", self._settings),
            "host_mass_msun": mass,
            "host_mass_definition": "200c",
            "host_mass_redshift": 0.0,
            "host_mass_z0": mass,
            "target_redshift": target,
            "accretion_mass_range_msun": [lo, hi],
            "accretion_mass_definition": "200c",
            "accretion_redshift_range": [zlo, zhi],
            "accretion_redshift_nodes": nodes.tolist(),
            "accretion_redshift_range_policy": policy,
            "host_history_mode": a["host_history_mode"],
            "deterministic_accretion_scatter_anchor_redshift": anchor,
            "deterministic_tidal_scatter_anchor_redshift": 1.0 if deterministic else None,
            "power_dependency_contract": "one particle mass/q10 law shared by sharp-k variance and top-hat concentration",
            "fixed_numerics": {
                "odeint_output_nodes": 100,
                "accretion_auxiliary_redshift_nodes": 1000,
                "concentration_mass_nodes": 100,
                "concentration_tophat_nodes": 500,
            },
            "solver_options": plain(s["solver_options"]),
            "survival": f"strict c_t > {d['ct_threshold']}",
            "ct_threshold": d["ct_threshold"],
        }
        preparation = _Preparation(nodes, s["solver_options"], d["ct_threshold"], metadata)
        execution = model._execute_population(parameters, preparation=preparation)
        return execution.to_catalog(model._catalog_metadata(parameters, preparation=preparation))
