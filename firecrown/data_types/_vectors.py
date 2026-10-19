"""Data and theory vector types.

This module contains wrapper classes for numpy arrays that represent observed data
and theoretical predictions.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

_NBit = TypeVar("_NBit", bound=npt.NBitBase)


def _vector_values(
    vals: npt.NDArray[np.integer[_NBit] | np.floating[_NBit]],
) -> npt.NDArray[np.float64]:
    """Validate vector ingress before conversion can discard masks or values."""
    if np.ma.isMaskedArray(vals):
        raise TypeError("Vector values must not be masked.")
    if vals.dtype.kind not in "iuf":
        raise TypeError("Vector values must have a real integer or floating dtype.")
    return np.asarray(vals, dtype=np.float64)


class DataVector(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """Wrapper for a np.ndarray that represents some observed data values."""

    @classmethod
    def create(
        cls, vals: npt.NDArray[np.integer[_NBit] | np.floating[_NBit]]
    ) -> DataVector:
        """Create a DataVector that wraps float64 values from the given array.

        Conforming arrays share storage, including strided arrays. Converting
        other real dtypes allocates storage and may lose precision; widening
        float32 does not recover prior rounding. Boolean, complex, string,
        object, structured, and masked arrays are rejected.

        :param vals: the real integer or floating array to convert and wrap
        :returns: a new DataVector
        """
        return _vector_values(vals).view(cls)

    @classmethod
    def from_list(cls, vals: list[float]) -> DataVector:
        """Create a DataVector from the given list of floats.

        :param vals: the list of floats
        :returns: a new DataVector
        """
        array = np.array(vals)
        return cls.create(array)


class TheoryVector(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """Wrapper for an np.ndarray that represents a prediction by some theory."""

    @classmethod
    def create(
        cls, vals: npt.NDArray[np.integer[_NBit] | np.floating[_NBit]]
    ) -> TheoryVector:
        """Create a TheoryVector that wraps float64 values from the given array.

        Conforming arrays share storage, including strided arrays. Converting
        other real dtypes allocates storage and may lose precision; widening
        float32 does not recover prior rounding. Boolean, complex, string,
        object, structured, and masked arrays are rejected.

        :param vals: the real integer or floating array to convert and wrap
        :returns: a new TheoryVector
        """
        return _vector_values(vals).view(cls)

    @classmethod
    def from_list(cls, vals: list[float]) -> TheoryVector:
        """Create a TheoryVector from the given list of floats.

        :param vals: the list of floats
        :returns: a new TheoryVector
        """
        array = np.array(vals)
        return cls.create(array)
