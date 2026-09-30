"""Native WDM contracts against independent pre-edit installed results."""

import hashlib
import json
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from itamae.types import WeightedSubhaloCatalog

from sashimi_w import WDM, Subhalos

ROOT = Path(__file__).parent / "references/native-api-baseline"


def configured(name="default"):
    record = json.loads((ROOT / (name + ".json")).read_text())
    p, constructor = record["parameters"], record["constructor"]
    groups = {
        "accretion": {
            "mass_nodes": p["N_ma"],
            "redshift_step": p["dz"],
            "host_history_nodes": p["N_hermNa"],
        },
        "concentration": {"scatter_dex": p["sigmalogc"], "quadrature_nodes": p["N_herm"]},
    }
    groups["dark_matter"] = {"mass_keV": constructor["mass_wdm"]}
    groups["accretion"].update(
        host_history_mode="deterministic" if p["N_hermNa"] == 1 else "quadrature",
        sigmafac=p["sigmafac"],
    )
    groups["stripping"] = {"profile_change": p["profile_change"]}
    request = {
        "host_mass_msun": p["M0"],
        "redshift": p["redshift"],
        "accretion_mass_range_msun": (10 ** p["logmamin"], 10 ** p["logmamax"]),
        "accretion_redshift_range": (p["redshift"], p["zmax"]),
    }
    return WDM().configure(**groups), request, record


def as_pair(result):
    return (
        result
        if isinstance(result, dict) or not hasattr(result, "columns")
        else {"default": result}
    )


@pytest.mark.parametrize("name", [p.stem for p in ROOT.glob("*.json")])
def test_native_and_legacy_match_independent_baseline(name):
    model, request, record = configured(name)
    assert record["source"] == "dcef1910d42cab940567be448ffc79b42d436802"
    path = ROOT / (name + ".npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    native = as_pair(model.population(**request))
    legacy = {
        "default": Subhalos(**record["constructor"]).rs_rhos_catalog_calc(**record["parameters"])
    }
    with np.load(path) as saved:
        for state, catalog in native.items():
            for key, array in {**catalog.columns, **catalog.weights}.items():
                expected = saved[state + "__" + key]
                if array.dtype.kind == "b":
                    np.testing.assert_array_equal(array, expected)
                else:
                    np.testing.assert_allclose(array, expected, rtol=5e-12, atol=1e-300)
            for key, array in catalog.columns.items():
                np.testing.assert_array_equal(array, legacy[state].columns[key])
            for key, array in catalog.weights.items():
                np.testing.assert_array_equal(array, legacy[state].weights[key])


def test_configuration_is_detached_immutable_and_partial():
    model, _, _ = configured()
    values = {"scatter_dex": 0.23}
    next_model = model.configure(concentration=values)
    values["scatter_dex"] = 999
    assert next_model.resolved_settings["concentration"]["scatter_dex"] == 0.23
    assert model.resolved_settings["concentration"]["scatter_dex"] == 0.128
    assert next_model.resolved_settings["accretion"] == model.resolved_settings["accretion"]
    with pytest.raises(TypeError):
        next_model.resolved_settings["concentration"]["scatter_dex"] = 0
    with pytest.raises(FrozenInstanceError):
        next_model._settings = {}


def test_repeated_runs_do_not_reuse_mutable_population_state():
    model, request, _ = configured()
    first = as_pair(model.population(**request))
    saved = {state: {k: v.copy() for k, v in cat.columns.items()} for state, cat in first.items()}
    model.configure(concentration={"scatter_dex": 0.2}).population(
        **{**request, "host_mass_msun": 2e10}
    )
    again = as_pair(model.population(**request))
    for state in saved:
        for key, value in saved[state].items():
            np.testing.assert_array_equal(value, first[state].columns[key])
            np.testing.assert_array_equal(value, again[state].columns[key])
    assert not hasattr(model, "catalog") and not hasattr(model, "M0")


def test_executed_provenance_and_weights_roundtrip(tmp_path):
    model, request, _ = configured()
    products = as_pair(model.population(**request))
    for state, catalog in products.items():
        assert (
            catalog.metadata["resolved_settings"]["accretion"]["mass_nodes"]
            == model.resolved_settings["accretion"]["mass_nodes"]
        )
        assert catalog.metadata["host_mass_redshift"] == 0.0
        path = tmp_path / (state + ".npz")
        catalog.to_npz(path)
        restored = WeightedSubhaloCatalog.from_npz(path)
        assert restored.metadata == catalog.metadata
        for key in catalog.weights:
            np.testing.assert_array_equal(restored.weights[key], catalog.weights[key])
        np.testing.assert_array_equal(
            catalog.weight_final, np.prod(list(catalog.weights.values()), axis=0)
        )


@pytest.mark.parametrize(
    "options",
    [
        {"rtol": 0.0},
        {"h0": 0.1},
        {"mxordn": 0},
        {"mxords": 6},
        {"hmin": 0.2, "hmax": 0.1},
        {"Dfun": lambda x: x},
        {"tfirst": True},
    ],
)
def test_bad_ode_controls_rejected_before_execution(options):
    model, _, _ = configured()
    with pytest.raises((ValueError, TypeError)):
        model.configure(stripping={"solver": "odeint", "solver_options": options})


def test_unknown_and_unsupported_component_settings_fail_early():
    model, _, _ = configured()
    for group in (
        {"concentration": {"relation": lambda x: x}},
        {"accretion": {"mass_nodes": True}},
        {"stripping": {"solver": "unknown"}},
        {"disruption": {"ct_threshold": -1}},
        {"concentration": {"prescription": "unknown"}},
        {"stripping": {"ignored": 1}},
    ):
        with pytest.raises((ValueError, TypeError)):
            model.configure(**group)


def test_explicit_off_grid_redshift_support_is_bounded():
    model, request, _ = configured()
    catalog = next(
        iter(
            as_pair(
                model.population(
                    **{**request, "redshift": 0.1, "accretion_redshift_range": (0.2, 1.3)}
                )
            ).values()
        )
    )
    assert np.max(catalog.columns["z_acc"]) <= 1.3
    np.testing.assert_allclose(np.unique(catalog.columns["z_acc"]), [0.7, 1.2], rtol=0, atol=2e-16)
    assert catalog.metadata["accretion_redshift_range_policy"] == "explicit-bounded"


def test_mode_transitions_require_coherent_explicit_settings():
    model, _, _ = configured()
    with pytest.raises(ValueError, match="set both explicitly"):
        model.configure(accretion={"host_history_mode": "deterministic"})
    deterministic = model.configure(
        accretion={"host_history_mode": "deterministic", "host_history_nodes": 1, "sigmafac": 0.5}
    )
    with pytest.raises(ValueError, match="reset inherited"):
        deterministic.configure(accretion={"host_history_mode": "quadrature"})
    with pytest.raises(ValueError, match="sigmafac"):
        deterministic.configure(
            accretion={"host_history_mode": "quadrature", "host_history_nodes": 3}
        )
    returned = deterministic.configure(
        accretion={"host_history_mode": "quadrature", "host_history_nodes": 3, "sigmafac": 0.0}
    )
    assert returned.resolved_settings["accretion"] == model.resolved_settings["accretion"]


def test_deterministic_scatter_anchors_are_explicit_and_nonsquare_supported():
    model, request, _ = configured()
    model = model.configure(
        accretion={"host_history_mode": "deterministic", "host_history_nodes": 1, "sigmafac": 0.0}
    )
    with pytest.raises(ValueError, match="sampled z>1"):
        model.population(**request)
    catalog = model.population(**{**request, "accretion_redshift_range": (0.2, 1.8)})
    assert len(catalog.columns["m_bound"]) == 3 * 4 * 2
    assert catalog.metadata["deterministic_accretion_scatter_anchor_redshift"] == 1.2
    assert catalog.metadata["deterministic_tidal_scatter_anchor_redshift"] == 1.0
    assert catalog.metadata["host_history_mode"] == "deterministic"


def test_host_reference_epoch_unsupported_before_physics(monkeypatch):
    model, request, _ = configured()
    import sashimi_w._itamae_migration as migration

    def forbidden(*args, **kwargs):
        raise AssertionError("No model should be created")

    monkeypatch.setattr(migration, "Subhalos", forbidden)
    with pytest.raises(ValueError, match="host_mass_redshift=0"):
        model.population(**{**request, "host_mass_redshift": 0.5})


def test_particle_mass_changes_both_variance_measure_and_concentration():
    model, request, _ = configured()
    first = model.population(**request)
    changed = model.configure(dark_matter={"mass_keV": 4.0}).population(**request)
    assert first.metadata["wdm_power_q"] == changed.metadata["wdm_power_q"] == 10.0
    assert first.metadata["power_identifier"] != changed.metadata["power_identifier"]
    assert first.metadata["variance_identifier"] != changed.metadata["variance_identifier"]
    assert not np.array_equal(first.columns["r_s_acc"], changed.columns["r_s_acc"])
    assert not np.array_equal(first.weights["weight_base"], changed.weights["weight_base"])
    assert first.metadata["cosmology_parameters"] == {
        "omega_m0": 0.27,
        "omega_lambda0": 0.73,
        "h": 0.7,
    }


def test_native_ode_controls_and_survival_reach_existing_components(monkeypatch):
    import sashimi_w._itamae_components as components
    from scipy.integrate import solve_ivp

    calls = []
    original = components.solve_evolution

    def checked(rhs, initial, grid, **kwargs):
        calls.append(kwargs.copy())
        actual = original(rhs, initial, grid, **kwargs)
        reference = solve_ivp(
            lambda z, m: rhs(z, m),
            (grid[0], grid[-1]),
            initial,
            method="DOP853",
            rtol=1e-11,
            atol=1e-10,
        )
        assert reference.success
        np.testing.assert_allclose(actual[-1], reference.y[:, -1], rtol=2e-7)
        return actual

    monkeypatch.setattr(components, "solve_evolution", checked)
    model, request, _ = configured()
    # This interval does not cross a nearest-host-mass concentration-grid knot;
    # unrestricted DOP853 is not a certified reference across those jumps.
    model = model.configure(accretion={"redshift_step": 0.1})
    request = {**request, "redshift": 0.1, "accretion_redshift_range": (0.1, 0.3)}
    result = model.configure(
        stripping={"solver_options": {"rtol": 1e-9, "atol": 1e-11, "hmax": 0.1}},
        disruption={"ct_threshold": 2.0},
    ).population(**request)
    assert calls and all(
        call["rtol"] == 1e-9 and call["atol"] == 1e-11 and call["odeint_options"]["hmax"] == 0.1
        for call in calls
    )
    np.testing.assert_array_equal(result.columns["survive"], result.columns["c_t"] > 2.0)
    np.testing.assert_array_equal(result.weights["weight_survival"], result.columns["survive"])
    assert result.metadata["ct_threshold"] == 2.0
    assert "explicit-controls" in result.metadata["solver_identifier"]


@pytest.mark.parametrize(
    "override",
    [
        {"host_mass_msun": 0.0},
        {"host_mass_msun": True},
        {"host_mass_definition": "vir"},
        {"redshift": float("nan")},
        {"accretion_mass_range_msun": (10.0, 1.0)},
        {"accretion_redshift_range": (0.0, 0.1)},
    ],
)
def test_invalid_problem_inputs_are_rejected(override):
    model, request, _ = configured()
    with pytest.raises((ValueError, TypeError)):
        model.population(**{**request, **override})


def test_omitted_redshift_domain_preserves_legacy_support():
    model, request, record = configured()
    request.pop("accretion_redshift_range")
    with warnings.catch_warnings(record=True) as native_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        native = as_pair(model.population(**request))
    parameters = {**record["parameters"], "zmax": 7.0}
    with warnings.catch_warnings(record=True) as legacy_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        legacy = {"default": Subhalos(**record["constructor"]).rs_rhos_catalog_calc(**parameters)}
    assert [(w.category.__name__, str(w.message)) for w in native_warnings] == [
        (w.category.__name__, str(w.message)) for w in legacy_warnings
    ]
    for catalog in native.values():
        assert all(np.all(np.isfinite(array)) for array in catalog.columns.values())
        assert np.all(np.isfinite(catalog.weight_final))
    for state, catalog in native.items():
        assert catalog.metadata["accretion_redshift_range_policy"] == "legacy-default-grid"
        for key, array in catalog.columns.items():
            np.testing.assert_array_equal(array, legacy[state].columns[key])
        for key, array in catalog.weights.items():
            np.testing.assert_array_equal(array, legacy[state].weights[key])
