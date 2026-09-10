"""Standard WDM API, immutable references and physical catalog contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import nfw_mass_function
from itamae.provenance import CALCULATION_METADATA_KEYS
from itamae.types import WeightedSubhaloCatalog
from itamae.units import AstropyUnits, NativeUnits
from sashimi_w import Subhalos, subhalos, STANDARD_T2_Q10, OmegaM, h, Msolar
import sashimi_w_itamae


ROOT = Path(__file__).parent
PARAMETERS = json.loads((ROOT / "golden/wdm_small_catalog.json").read_text())["parameters"]
PARAMETERS = {key: value for key, value in PARAMETERS.items() if key != "mass_wdm"}
COLUMNS = ["m200_acc", "z_acc", "r_s_acc", "rho_s_acc", "m_bound", "r_s", "rho_s", "c_t"]
SCALES = [1.0, 1.0, 1e-3, 1e18, 1.0, 1e-3, 1e18, 1.0]


@pytest.fixture(scope="module")
def population():
    q, convention = 10, STANDARD_T2_Q10
    model = Subhalos(2.0, wdm_power_convention=convention)
    return q, model, model.rs_rhos_catalog_calc(**PARAMETERS)


def test_standard_import_and_removed_mode():
    assert subhalos is Subhalos is sashimi_w_itamae.subhalos
    assert Subhalos(2).wdm_power_q == 10
    for mode in ("legacy", "consistent"):
        with pytest.raises(TypeError, match="physics_mode"):
            Subhalos(2, wdm_power_convention=STANDARD_T2_Q10, physics_mode=mode)


def test_frozen_records_keep_original_mode_and_source_identity():
    for name in ("wdm_small_catalog.json", "wdm_q10_small_catalog.json"):
        fixture = json.loads((ROOT / "golden" / name).read_text())
        provenance = fixture["provenance"]
        assert "physics_modes" in provenance
        assert len(provenance["generated_repository_revision"]) == 40
        assert len(provenance["itamae_source_revision"]) == 40
        assert provenance["fixture_schema"] == "sashimi-family:golden-provenance:v1"


def test_full_catalog_matches_independent_reference(population):
    q, model, catalog = population
    path = ROOT / "references" / f"B-all-q{q}.npz"
    provenance = json.loads(path.with_suffix(".json").read_text())
    assert provenance["role"] == "B"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["artifact_sha256"]
    with np.load(path) as expected:
        assert catalog.shape == (16,)
        for index, (name, scale) in enumerate(zip(COLUMNS, SCALES, strict=True)):
            np.testing.assert_allclose(
                catalog.columns[name], expected[f"tuple_{index}"] * scale, rtol=5e-10, atol=1e-14
            )
        np.testing.assert_array_equal(catalog.columns["survive"], expected["tuple_9"])
        np.testing.assert_allclose(
            catalog.weights["weight_base"] * catalog.weights["weight_concentration"],
            expected["tuple_8"],
            rtol=5e-12,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            catalog.weight_final, expected["tuple_8"] * expected["tuple_9"], rtol=5e-12, atol=1e-18
        )
    assert np.all(catalog.weight_final >= 0)
    assert np.all(catalog.columns["m_bound"] <= catalog.columns["m200_acc"])
    np.testing.assert_array_equal(catalog.columns["survive"], catalog.columns["c_t"] > 0.77)
    mass = (
        4
        * np.pi
        * catalog.columns["rho_s"]
        * catalog.columns["r_s"] ** 3
        * nfw_mass_function(catalog.columns["c_t"])
    )
    np.testing.assert_allclose(mass, catalog.columns["m_bound"], rtol=3e-10)
    concentration = catalog.weights["weight_concentration"].reshape(2, 2, 4)
    np.testing.assert_allclose(np.sum(concentration, axis=1), 1.0, rtol=2e-15)
    base = catalog.weights["weight_base"].reshape(2, 2, 4)
    np.testing.assert_array_equal(base[:, 0], base[:, 1])


def test_calculation_metadata_and_serialization(population, tmp_path):
    q, model, catalog = population
    assert set(CALCULATION_METADATA_KEYS) <= set(catalog.metadata)
    assert "physics_mode" not in catalog.metadata
    assert (
        catalog.metadata["calculation_specification"] == "sashimi-w:thermal-wdm-q10:2026-09-11:v2"
    )
    assert catalog.metadata["calculation_parameters"] == PARAMETERS
    assert catalog.metadata["wdm_power_q"] == q
    assert catalog.metadata["variance_identifier"] == model.variance_model().identifier
    assert catalog.metadata["power_identifier"] == model.variance_model().power.identifier
    assert catalog.metadata["variance_growth_power"] == 2
    assert catalog.metadata["cosmology_parameters"] == {
        "omega_m0": float(OmegaM),
        "omega_lambda0": 1 - float(OmegaM),
        "h": float(h),
    }
    for field in ("itamae_source_revision", "sashimi_source_revision"):
        assert len(catalog.metadata[field]) == 40
    path = tmp_path / f"q{q}.npz"
    catalog.to_npz(path)
    restored = WeightedSubhaloCatalog.from_npz(path)
    for name in catalog.columns:
        np.testing.assert_array_equal(catalog.columns[name], restored.columns[name])
    np.testing.assert_array_equal(catalog.weight_final, restored.weight_final)
    assert dict(catalog.metadata) == dict(restored.metadata)


def test_tuple_is_only_a_format_conversion(population, monkeypatch):
    _, model, catalog = population
    monkeypatch.setattr(model, "rs_rhos_catalog_calc", lambda **kw: catalog)
    result = model.rs_rhos_calc(**PARAMETERS)
    converted = model.catalog_from_tuple(
        result,
        weight_base=catalog.weights["weight_base"],
        weight_concentration=catalog.weights["weight_concentration"],
    )
    for name in catalog.columns:
        np.testing.assert_allclose(converted.columns[name], catalog.columns[name], rtol=3e-16)
    np.testing.assert_array_equal(converted.weight_final, catalog.weight_final)


def test_native_astropy_tuple_units(population):
    _, _, catalog = population
    raw = tuple(catalog.columns[name] / scale for name, scale in zip(COLUMNS, SCALES, strict=True))
    raw += (
        catalog.weights["weight_base"] * catalog.weights["weight_concentration"],
        catalog.columns["survive"],
    )
    first = Subhalos(
        2, wdm_power_convention=STANDARD_T2_Q10, unit_backend=NativeUnits()
    ).catalog_from_tuple(raw)
    second = Subhalos(
        2, wdm_power_convention=STANDARD_T2_Q10, unit_backend=AstropyUnits()
    ).catalog_from_tuple(raw)
    for name in first.columns:
        np.testing.assert_array_equal(first.columns[name], second.columns[name])
    np.testing.assert_array_equal(first.weight_final, second.weight_final)


def test_growth_derivative_and_physical_mass_boundary():
    model = Subhalos(2)
    assert model.growthD(0.0) == 1.0
    z = np.array([0.0, 0.5, 1.0, 3.0, 5.0])
    step = 1e-5
    numerical = (model.growthD(z + step) - model.growthD(z - step)) / (2 * step)
    np.testing.assert_allclose(model.dDdz(z), numerical, rtol=3e-9)
    assert model.linear_growth_factor(OmegaM, 1 - OmegaM, [0.5, 3.0]) == pytest.approx(
        model.growthD(3.0) / model.growthD(0.5)
    )
    scalar = model.conc200(1e9 * Msolar, 0.5)
    np.testing.assert_allclose(model.conc200(np.array([1e9]) * Msolar, 0.5), scalar, rtol=1e-14)


@pytest.mark.parametrize(
    "parameters,error",
    [
        ({"mass_wdm": 0}, "mass_wdm"),
        ({"mass_wdm": np.inf}, "mass_wdm"),
        ({"wdm_power_convention": "invalid"}, "wdm_power_convention"),
        ({"cosmology_backend": NativeFlatLCDM(omega_m0=0.3, h=0.7)}, "OmegaM"),
        ({"cosmology_backend": NativeFlatLCDM(omega_m0=0.27, h=0.6)}, "h="),
    ],
)
def test_constructor_errors(parameters, error):
    arguments = {"mass_wdm": 2.0, "wdm_power_convention": STANDARD_T2_Q10, **parameters}
    with pytest.raises(ValueError, match=error):
        Subhalos(**arguments)


@pytest.mark.parametrize(
    "invalid",
    [
        {"M0": 0},
        {"dz": 0},
        {"zmax": 0},
        {"redshift": -1},
        {"N_ma": 1},
        {"N_herm": 0},
        {"N_hermNa": False},
        {"logmamax": 4},
        {"profile_change": "yes"},
    ],
)
def test_catalog_rejects_invalid_inputs_before_work(invalid, monkeypatch):
    model = Subhalos(2.0, wdm_power_convention=STANDARD_T2_Q10)

    def forbidden(*args, **kwargs):
        raise AssertionError("numerical calculation must not run")

    monkeypatch.setattr(model, "_execute_population", forbidden)
    with pytest.raises((ValueError, TypeError)):
        model.rs_rhos_catalog_calc(**{**PARAMETERS, **invalid})
