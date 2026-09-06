"""Tests for updatable utility functions."""

from firecrown.updatable import ParamsMap, get_default_params, get_default_params_map
from tests.updatable.updatable_test_support import MinimalUpdatable, SimpleUpdatable


def test_get_default_params_empty_args():
    """Test get_default_params() with no arguments.

    Should return an empty dictionary (lines 19-23 in _utils.py).
    """
    result = get_default_params()
    assert result == {}
    assert isinstance(result, dict)


def test_get_default_params_map_empty_args():
    """Test get_default_params_map() with no arguments.

    Should return an empty ParamsMap (lines 32-33 in _utils.py).
    """
    result = get_default_params_map()
    assert isinstance(result, ParamsMap)
    assert len(result.keys()) == 0


def test_get_default_params_with_multiple_updatables():
    """Test get_default_params() with multiple updatables.

    Verifies that all default values are collected correctly.
    """
    obj1 = SimpleUpdatable()
    obj2 = MinimalUpdatable()

    # Test get_default_params
    result = get_default_params(obj1, obj2)
    assert result == {"x": 2.0, "y": 3.0, "a": 1.0}

    # Test get_default_params_map
    params_map = get_default_params_map(obj1, obj2)
    assert isinstance(params_map, ParamsMap)
    assert params_map.get("x") == 2.0
    assert params_map.get("y") == 3.0
    assert params_map.get("a") == 1.0
