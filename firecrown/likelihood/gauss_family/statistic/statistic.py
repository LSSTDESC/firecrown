"""

Gaussian Family Statistic Module
================================

The Statistic class describing objects that implement methods to compute the
data and theory vectors for a GaussianFamily subclass.

"""

from __future__ import annotations
from typing import List
from dataclasses import dataclass
import warnings
import numpy as np
import pyccl
import sacc

from ....updatable import Updatable
from .source.source import Systematic


class DataVector(np.ndarray):
    """This class wraps a np.ndarray that represents some observed data values."""

    @classmethod
    def create(cls, vals: np.ndarray) -> DataVector:
        """Create a DataVector that wraps a copy of the given array vals."""
        return vals.view(cls)

    @classmethod
    def from_list(cls, vals: List[float]) -> DataVector:
        """Create a DataVector from the given list of floats."""
        array = np.array(vals)
        return cls.create(array)


class TheoryVector(np.ndarray):
    """This class wraps a np.ndarray that represents an observation predicted by
    some theory."""

    @classmethod
    def create(cls, vals: np.ndarray) -> TheoryVector:
        """Create a TheoryVector that wraps a copy of the given array vals."""
        return vals.view(cls)

    @classmethod
    def from_list(cls, vals: List[float]) -> TheoryVector:
        """Create a TheoryVector from the given list of floats."""
        array = np.array(vals)
        return cls.create(array)


def residuals(data: DataVector, theory: TheoryVector) -> np.ndarray:
    """Return a bare np.ndarray with the difference between `data` and `theory`.
    This is to be preferred to using arithmetic on the vectors directly."""
    assert isinstance(data, DataVector)
    assert isinstance(theory, TheoryVector)
    return (data - theory).view(np.ndarray)


@dataclass
class StatisticsResult:
    """This is the type returned by the `compute` method of any `Statistic`."""

    data: DataVector
    theory: TheoryVector

    def __post_init__(self):
        """Make sure the data and theory vectors are of the same shape."""
        assert self.data.shape == self.theory.shape

    def residuals(self) -> np.ndarray:
        """Return the residuals -- the difference between data and theory."""
        return self.data - self.theory

    def __iter__(self):
        """Iterate through the data members. This is to allow automatic unpacking, as
        if the StatisticsResult were a tuple of (data, theory).

        This method is a temporary measure to help code migrate to the newer,
        safer interface for Statistic.compute()."""
        warnings.warn(
            "Iteration and tuple unpacking for StatisticsResult is "
            "deprecated.\nPlease use the StatisticsResult class accessors"
            ".data and .theory by name."
        )
        yield self.data
        yield self.theory


class Statistic(Updatable):
    """An abstract statistic class (e.g., two-point function, mass function, etc.)."""

    systematics: List[Systematic]
    sacc_indices: List[int]

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file."""

    def get_data_vector(self) -> DataVector:
        """Gets the statistic data vector."""
        raise NotImplementedError("Method `get_data_vector` is not implemented!")

    def compute_theory_vector(self, cosmo: pyccl.Cosmology) -> TheoryVector:
        """Compute a statistic from sources, applying any systematics."""
        raise NotImplementedError("Method `compute_theory_vector` is not implemented!")

    def compute(self, cosmo: pyccl.Cosmology) -> StatisticsResult:
        """Compute a statistic from sources, applying any systematics."""

        raise NotImplementedError("Method `compute` is not implemented!")
