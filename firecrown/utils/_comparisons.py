"""Comparison utilities for optional values."""

import numpy as np
from numpy import typing as npt


def compare_optional_arrays(x: None | npt.NDArray, y: None | npt.NDArray) -> bool:
    """Compare two arrays, allowing for either or both to be None.

    :param x: first array
    :param y: second array
    :return: whether the arrays are equal
    """
    if x is None and y is None:
        return True
    if x is not None and y is not None:
        return np.array_equal(x, y)
    # One is None and the other is not.
    return False


def compare_optionals(x: None | object, y: None | object) -> bool:
    """Compare two objects, allowing for either or both to be None.

    :param x: first object
    :param y: second object
    :return: whether the objects are equal
    """
    if x is None and y is None:
        return True
    if x is not None and y is not None:
        return x == y
    # One is None and the other is not.
    return False
