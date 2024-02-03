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


def test_named_parameters_set_from_basic_dict_string():
    params = NamedParameters()
    params.set_from_basic_dict({"a": "Im a string"})
    assert params.get_string("a") == "Im a string"

    params.set_from_basic_dict({"b": "Im another string"})
    assert params.get_string("a") == "Im a string"
    assert params.get_string("b") == "Im another string"


def test_named_parameters_set_from_basic_dict_int():
    params = NamedParameters()
    params.set_from_basic_dict({"a": 1})
    assert params.get_int("a") == 1

    params.set_from_basic_dict({"b": 2})
    assert params.get_int("a") == 1
    assert params.get_int("b") == 2


def test_named_parameters_set_from_basic_dict_float():
    params = NamedParameters()
    params.set_from_basic_dict({"a": 1.0})
    assert params.get_float("a") == 1.0

    params.set_from_basic_dict({"b": 2.0})
    assert params.get_float("a") == 1.0
    assert params.get_float("b") == 2.0


def test_named_parameters_set_from_basic_dict_bool():
    params = NamedParameters()
    params.set_from_basic_dict({"a": True})
    assert params.get_bool("a") is True

    params.set_from_basic_dict({"b": False})
    assert params.get_bool("a") is True
    assert params.get_bool("b") is False


def test_named_parameters_set_from_basic_dict_int_array():
    params = NamedParameters()
    params.set_from_basic_dict({"a": [1, 2, 3]})
    assert params.get_int_array("a").tolist() == [1, 2, 3]

    params.set_from_basic_dict({"b": [4, 5, 6]})
    assert params.get_int_array("a").tolist() == [1, 2, 3]
    assert params.get_int_array("b").tolist() == [4, 5, 6]


def test_named_parameters_set_from_basic_dict_float_array():
    params = NamedParameters()
    params.set_from_basic_dict({"a": [1.0, 2.0, 3.0]})
    assert params.get_float_array("a").tolist() == [1.0, 2.0, 3.0]

    params.set_from_basic_dict({"b": [4.0, 5.0, 6.0]})
    assert params.get_float_array("a").tolist() == [1.0, 2.0, 3.0]
    assert params.get_float_array("b").tolist() == [4.0, 5.0, 6.0]


def test_named_parameters_set_from_basic_dict_mixed():
    params = NamedParameters()
    params.set_from_basic_dict(
        {
            "a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "c": "Im a string",
            "d": 1,
            "e": 1.0,
            "f": True,
        }
    )
    assert params.get_int_array("a").tolist() == [1, 2, 3]
    assert params.get_float_array("b").tolist() == [1.0, 2.0, 3.0]
    assert params.get_string("c") == "Im a string"
    assert params.get_int("d") == 1
    assert params.get_float("e") == 1.0
    assert params.get_bool("f") is True


def test_named_parameters_convert_to_basic_dict_string():
    params = NamedParameters({"a": "Im a string"})
    assert params.convert_to_basic_dict() == {"a": "Im a string"}

    params = NamedParameters({"a": "Im a string", "b": "Im another string"})
    assert params.convert_to_basic_dict() == {
        "a": "Im a string",
        "b": "Im another string",
    }


def test_named_parameters_convert_to_basic_dict_int():
    params = NamedParameters({"a": 1})
    assert params.convert_to_basic_dict() == {"a": 1}

    params = NamedParameters({"a": 1, "b": 2})
    assert params.convert_to_basic_dict() == {"a": 1, "b": 2}


def test_named_parameters_convert_to_basic_dict_float():
    params = NamedParameters({"a": 1.0})
    assert params.convert_to_basic_dict() == {"a": 1.0}

    params = NamedParameters({"a": 1.0, "b": 2.0})
    assert params.convert_to_basic_dict() == {"a": 1.0, "b": 2.0}


def test_named_parameters_convert_to_basic_dict_bool():
    params = NamedParameters({"a": True})
    assert params.convert_to_basic_dict() == {"a": True}

    params = NamedParameters({"a": True, "b": False})
    assert params.convert_to_basic_dict() == {"a": True, "b": False}


def test_named_parameters_convert_to_basic_dict_int_array():
    params = NamedParameters({"a": np.array([1, 2, 3])})
    assert params.convert_to_basic_dict() == {"a": [1, 2, 3]}

    params = NamedParameters({"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])})
    assert params.convert_to_basic_dict() == {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }


def test_named_parameters_convert_to_basic_dict_float_array():
    params = NamedParameters({"a": np.array([1.0, 2.0, 3.0])})
    assert params.convert_to_basic_dict() == {"a": [1.0, 2.0, 3.0]}

    params = NamedParameters(
        {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}
    )
    assert params.convert_to_basic_dict() == {
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
    }


def test_named_parameters_convert_to_basic_dict_mixed():
    params = NamedParameters(
        {
            "a": np.array([1, 2, 3]),
            "b": np.array([1.0, 2.0, 3.0]),
            "c": "Im a string",
            "d": 1,
            "e": 1.0,
            "f": True,
        }
    )
    assert params.convert_to_basic_dict() == {
        "a": [1, 2, 3],
        "b": [1.0, 2.0, 3.0],
        "c": "Im a string",
        "d": 1,
        "e": 1.0,
        "f": True,
    }


def test_named_parameters_to_set():
    params = NamedParameters({"a": True})
    assert params.to_set() == {"a"}

    params = NamedParameters({"a": True, "b": False})
    assert params.to_set() == {"a", "b"}

    params = NamedParameters({"a": True, "b": False, "c": True})
    assert params.to_set() == {"a", "b", "c"}


def test_invalid_set_from_basic_dict():
    params = NamedParameters()
    with pytest.raises(ValueError):
        params.set_from_basic_dict({"a": {"b": "c"}})  # type: ignore


def test_invalid_set_from_basic_dict_sequence():
    params = NamedParameters()
    with pytest.raises(ValueError):
        params.set_from_basic_dict({"a": ["a", "b", "c"]})  # type: ignore


def test_invalid_convert_to_basic_dict():
    params = NamedParameters({"a": {"b": "c"}})  # type: ignore
    with pytest.raises(ValueError):
        params.convert_to_basic_dict()


def test_invalid_convert_to_basic_dict_sequence():
    params = NamedParameters({"a": ["a", "b", "c"]})  # type: ignore
    with pytest.raises(ValueError):
        params.convert_to_basic_dict()
