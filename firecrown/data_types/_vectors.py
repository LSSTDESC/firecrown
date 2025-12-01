"""Data and theory vector types.

This module contains wrapper classes for numpy arrays that represent observed data
and theoretical predictions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class DataVector(npt.NDArray[np.float64]):
    """Wrapper for a np.ndarray that represents some observed data values."""

    @classmethod
    def create(cls, vals: npt.NDArray[np.float64]) -> DataVector:
        """Create a DataVector that wraps a copy of the given array vals.

        :param vals: the array to be copied and wrapped
        :return: a new DataVector
        """
        return vals.view(cls)

    @classmethod
    def from_list(cls, vals: list[float]) -> DataVector:
        """Create a DataVector from the given list of floats.

        :param vals: the list of floats
        :return: a new DataVector
        """
        array = np.array(vals)
        return cls.create(array)


class TheoryVector(npt.NDArray[np.float64]):
    """Wrapper for an np.ndarray that represents a prediction by some theory."""

    @classmethod
    def create(cls, vals: npt.NDArray[np.float64]) -> TheoryVector:
        """Create a TheoryVector that wraps a copy of the given array vals.

        :param vals: the array to be copied and wrapped
        :return: a new TheoryVector
        """
        return vals.view(cls)

    @classmethod
    def from_list(cls, vals: list[float]) -> TheoryVector:
        """Create a TheoryVector from the given list of floats.

        :param vals: the list of floats
        :return: a new TheoryVector
        """
        array = np.array(vals)
        return cls.create(array)
