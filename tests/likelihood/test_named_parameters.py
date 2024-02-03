"""Tests for the class firecrown.likelihood.NamedParameters."""

import pytest
import numpy as np

from firecrown.likelihood.likelihood import NamedParameters


def test_named_parameters_sanity():
    params = NamedParameters({"a": True})
    assert params.get_bool("a") is True

    params = NamedParameters({"a": "Im a string"})
    assert params.get_string("a") == "Im a string"

    params = NamedParameters({"a": 1})
    assert params.get_int("a") == 1

    params = NamedParameters({"a": 1.0})
    assert params.get_float("a") == 1.0

    params = NamedParameters({"a": np.array([1, 2, 3])})
    assert params.get_int_array("a").tolist() == [1, 2, 3]

    params = NamedParameters({"a": np.array([1.0, 2.0, 3.0])})
    assert params.get_float_array("a").tolist() == [1.0, 2.0, 3.0]

    params = NamedParameters({"a": False})
    assert params.to_set() == {"a"}


def test_named_parameters_default():
    params = NamedParameters({})

    assert params.get_bool("a", True) is True
    assert params.get_string("b", "Im a string") == "Im a string"
    assert params.get_int("c", 1) == 1
    assert params.get_float("d", 1.0) == 1.0

    params = NamedParameters({"a": False, "b": "Im another string", "c": 2, "d": 2.2})

    assert params.get_bool("a", True) is False
    assert params.get_string("b", "Im a string") == "Im another string"
    assert params.get_int("c", 1) == 2
    assert params.get_float("d", 1.0) == 2.2


def test_named_parameters_wrong_type_bool():
    params = NamedParameters({"a": True})
    with pytest.raises(AssertionError):
        params.get_string("a")

    # Bools are ints in python
    # with pytest.raises(AssertionError):
    #    params.get_int("a")

    with pytest.raises(AssertionError):
        params.get_float("a")

    with pytest.raises(AssertionError):
        params.get_int_array("a")

    with pytest.raises(AssertionError):
        params.get_float_array("a")


def test_named_parameters_wrong_type_string():
    params = NamedParameters({"a": "Im a string"})
    with pytest.raises(AssertionError):
        params.get_bool("a")

    with pytest.raises(AssertionError):
        params.get_int("a")

    with pytest.raises(AssertionError):
        params.get_float("a")

    with pytest.raises(AssertionError):
        params.get_int_array("a")

    with pytest.raises(AssertionError):
        params.get_float_array("a")


def test_named_parameters_wrong_type_int():
    params = NamedParameters({"a": 1})
    with pytest.raises(AssertionError):
        params.get_bool("a")

    with pytest.raises(AssertionError):
        params.get_string("a")

    with pytest.raises(AssertionError):
        params.get_float("a")

    with pytest.raises(AssertionError):
        params.get_int_array("a")

    with pytest.raises(AssertionError):
        params.get_float_array("a")


def test_named_parameters_wrong_type_float():
    params = NamedParameters({"a": 1.0})
    with pytest.raises(AssertionError):
        params.get_bool("a")

    with pytest.raises(AssertionError):
        params.get_string("a")

    with pytest.raises(AssertionError):
        params.get_int("a")

    with pytest.raises(AssertionError):
        params.get_int_array("a")

    with pytest.raises(AssertionError):
        params.get_float_array("a")


def test_named_parameters_wrong_type_int_array():
    params = NamedParameters({"a": np.array([1, 2, 3])})
    with pytest.raises(AssertionError):
        params.get_bool("a")

    with pytest.raises(AssertionError):
        params.get_string("a")

    with pytest.raises(AssertionError):
        params.get_int("a")

    with pytest.raises(AssertionError):
        params.get_float("a")

    # Int arrays are float arrays in python
    # with pytest.raises(AssertionError):
    #    params.get_float_array("a")


def test_named_parameters_wrong_type_float_array():
    params = NamedParameters({"a": np.array([1.0, 2.0, 3.0])})
    with pytest.raises(AssertionError):
        params.get_bool("a")

    with pytest.raises(AssertionError):
        params.get_string("a")

    with pytest.raises(AssertionError):
        params.get_int("a")

    with pytest.raises(AssertionError):
        params.get_float("a")

    # Float arrays are int arrays in python
    # with pytest.raises(AssertionError):
    #    params.get_int_array("a")
