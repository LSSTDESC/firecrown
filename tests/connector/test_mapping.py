"""
Unit testing for the mapping module.
"""
from typing import Any, Dict
import pytest
import numpy as np
from firecrown.connector import mapping
from firecrown.connector.mapping import Mapping, mapping_builder
from firecrown.likelihood.likelihood import NamedParameters


# TODO: Refactor these test functions to use a sensible fixture.
def test_conversion_from_cosmosis_camb():
    cosmosis_params = {
        "a_s": 2.1e-09,
        "baryon_fraction": 0.13333333333333333,
        "cs2_de": 1.0,
        "h0": 0.72,
        "hubble": 72.0,
        "k": -0.0,
        "k_s": 0.05000000074505806,
        "n_run": 0.0,
        "n_s": 0.96,
        "ombh2": 0.020735999999999997,
        "omch2": 0.134784,
        "omega_b": 0.04,
        "omega_c": 0.26,
        "omega_k": 0.0,
        "omega_lambda": 0.7,
        "omega_m": 0.3,
        "omega_nu": 0.00624525169,
        "ommh2": 0.15552,
        "omnuh2": 0.0,
        "r_t": 0.0,
        "sigma_8": 0.9076407068565138,
        "tau": 0.08,
        "w": -1.0,
        "wa": 0.0,
        "yhe": 0.23999999463558197,
    }
    named_params = NamedParameters(cosmosis_params)
    p = mapping.mapping_builder(input_style="CosmoSIS")
    p.set_params_from_cosmosis(named_params)
    assert p.Omega_c == cosmosis_params["omega_c"]
    assert p.Omega_b == cosmosis_params["omega_b"]
    assert p.h == cosmosis_params["h0"]
    assert p.A_s is None
    assert p.sigma8 == cosmosis_params["sigma_8"]
    assert p.n_s == cosmosis_params["n_s"]
    assert p.Omega_k == cosmosis_params["omega_k"]
    assert p.Omega_g is None
    assert p.Neff == pytest.approx(3.046)
    assert p.m_nu == pytest.approx(0.3015443336635814)
    assert p.m_nu_type == "normal"  # Currently the only option
    assert p.w0 == cosmosis_params["w"]
    assert p.wa == cosmosis_params["wa"]
    assert p.T_CMB == 2.7255  # currently hard-wired


def test_conversion_from_cosmosis_camb_using_delta_neff():
    cosmosis_params = {
        "a_s": 2.1e-09,
        "baryon_fraction": 0.13333333333333333,
        "cs2_de": 1.0,
        "delta_neff": 0.125,
        "h0": 0.72,
        "hubble": 72.0,
        "k": -0.0,
        "k_s": 0.05000000074505806,
        "n_run": 0.0,
        "n_s": 0.96,
        "ombh2": 0.020735999999999997,
        "omch2": 0.134784,
        "omega_b": 0.04,
        "omega_c": 0.26,
        "omega_k": 0.0,
        "omega_lambda": 0.7,
        "omega_m": 0.3,
        "omega_nu": 0.0,
        "ommh2": 0.15552,
        "omnuh2": 0.0,
        "r_t": 0.0,
        "sigma_8": 0.9076407068565138,
        "tau": 0.08,
        "w": -1.0,
        "wa": 0.0,
        "yhe": 0.23999999463558197,
    }
    named_params = NamedParameters(cosmosis_params)
    p = mapping.mapping_builder(input_style="CosmoSIS")
    p.set_params_from_cosmosis(named_params)
    assert p.Neff == pytest.approx(3.171)


def test_get_params_names():
    fc_map = Mapping()

    with pytest.deprecated_call():
        params_names = fc_map.get_params_names()
        assert not params_names


def test_transform_k_h_to_k():
    fc_map = Mapping()

    with pytest.deprecated_call():
        fc_map.transform_k_h_to_k([])


def test_transform_p_k_h3_to_p_k():
    fc_map = Mapping()

    with pytest.deprecated_call():
        fc_map.transform_p_k_h3_to_p_k([])


def test_transform_h_to_h_over_h0():
    fc_map = Mapping()

    with pytest.deprecated_call():
        fc_map.transform_h_to_h_over_h0([])


def test_sigma8_and_A_s():
    fc_map = Mapping()

    params_dict: Dict[str, Any] = {
        "Omega_c": 0.26,
        "Omega_b": 0.04,
        "h": 0.72,
        "n_s": 0.96,
        "Omega_k": 0.0,
        "Neff": 3.046,
        "m_nu": 0.0,
        "m_nu_type": "normal",
        "w0": -1.0,
        "wa": 0.0,
        "T_CMB": 2.7255,
    }

    with pytest.raises(
        ValueError, match="Exactly one of A_s and sigma8 must be supplied"
    ):
        fc_map.set_params(
            **params_dict,
            sigma8=0.8,
            A_s=2.1e-9,
        )

    with pytest.raises(
        ValueError, match="Exactly one of A_s and sigma8 must be supplied"
    ):
        fc_map.set_params(**params_dict)


def test_mappping_builder():
    with pytest.raises(
        ValueError, match="input_style must be .* not invalid_input_style"
    ):
        mapping_builder(input_style="invalid_input_style")


def test_mapping_cosmosis():
    mapping_cosmosis = mapping_builder(input_style="CosmoSIS")
    assert isinstance(mapping_cosmosis, Mapping)

    assert mapping_cosmosis.get_params_names() == [
        "h0",
        "omega_b",
        "omega_c",
        "sigma_8",
        "n_s",
        "omega_k",
        "delta_neff",
        "omega_nu",
        "w",
        "wa",
    ]


def test_mapping_cosmosis_k_h_to_h(mapping_cosmosis):
    k_h_array = np.geomspace(0.1, 10.0, 10)
    k_array = mapping_cosmosis.transform_k_h_to_k(k_h_array)

    assert np.allclose(k_h_array * mapping_cosmosis.h, k_array)


def test_mapping_cosmosis_p_k_h3_to_p_k(mapping_cosmosis):
    p_k_h3_array = np.geomspace(0.1, 10.0, 10)
    p_k_array = mapping_cosmosis.transform_p_k_h3_to_p_k(p_k_h3_array)

    assert np.allclose(p_k_h3_array / mapping_cosmosis.h**3, p_k_array)
