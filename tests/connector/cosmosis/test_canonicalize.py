import pytest
import numpy as np

from firecrown.likelihood._base import NamedParameters
from firecrown.connector.cosmosis.likelihood import (
    _canonicalize_cosmological_parameters,
)


def test_canonicalize_scalars_and_ignore_non_numeric():
    named = NamedParameters(
        {
            "omega_c": 0.25,
            "omega_b": 0.05,
            "h0": 0.682,
            "sigma_8": 0.801,
            "nnu": 3.046,
            "mnu": 0.06,
            "TCMB": 2.7255,
            "consistency_module_was_used": True,
        }
    )

    res = _canonicalize_cosmological_parameters(named)

    assert isinstance(res["Omega_c"], float)
    assert res["Omega_c"] == pytest.approx(0.25)
    assert res["Omega_b"] == pytest.approx(0.05)
    assert res["h"] == pytest.approx(0.682)
    assert res["sigma8"] == pytest.approx(0.801)
    assert res["Neff"] == pytest.approx(3.046)
    assert res["m_nu"] == pytest.approx(0.06)
    assert res["T_CMB"] == pytest.approx(2.7255)

    # non-numeric flag should be ignored
    assert "consistency_module_was_used" not in res


def test_canonicalize_list_values():
    named = NamedParameters({"sigma_8": np.array([0.8, 0.81]), "mnu": np.array([0.06])})
    res = _canonicalize_cosmological_parameters(named)

    assert isinstance(res["sigma8"], list)
    assert res["sigma8"] == [0.8, 0.81]
    assert isinstance(res["m_nu"], list)
    assert res["m_nu"] == [0.06]


def test_unknown_keys_ignored():
    named = NamedParameters({"foo_bar": 1.23})
    res = _canonicalize_cosmological_parameters(named)
    assert res == {}


def test_list_with_non_numeric_elements_skipped():
    # sigma_8 contains a non-numeric entry -> conversion will raise and key skipped
    named = NamedParameters({"sigma_8": np.array([0.8, "bad"], dtype=object)})
    res = _canonicalize_cosmological_parameters(named)
    assert "sigma8" not in res


def test_scalar_non_numeric_skipped():
    # TCMB present but non-numeric string -> should be ignored
    named = NamedParameters({"TCMB": "not-a-number"})
    res = _canonicalize_cosmological_parameters(named)
    assert "T_CMB" not in res
