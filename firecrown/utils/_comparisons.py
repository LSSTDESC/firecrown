"""Comparison utilities for optional values."""

from typing import TypeVar

import numpy as np
from numpy import typing as npt

_ScalarT = TypeVar("_ScalarT", bound=np.generic)


def compare_optional_arrays(
    x: None | npt.NDArray[_ScalarT], y: None | npt.NDArray[_ScalarT]
) -> bool:
    """Compare two arrays, allowing for either or both to be None.

    :param x: first array
    :param y: second array
    :returns: whether the arrays are equal
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
    :returns: whether the objects are equal
    """
    if x is None and y is None:
        return True
    if x is not None and y is not None:
        return x == y
    # One is None and the other is not.
    return False
