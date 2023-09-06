"""

Gaussian Family Statistic Module
================================

The Statistic class describing objects that implement methods to compute the
data and theory vectors for a GaussianFamily subclass.

"""

from __future__ import annotations
from typing import List
from dataclasses import dataclass
from abc import abstractmethod
import warnings
import numpy as np
import numpy.typing as npt
import sacc

from ....modeling_tools import ModelingTools
from ....updatable import Updatable
from .source.source import SourceSystematic


class DataVector(npt.NDArray[np.float64]):
    """This class wraps a np.ndarray that represents some observed data values."""

    @classmethod
    def create(cls, vals: npt.NDArray[np.float64]) -> DataVector:
        """Create a DataVector that wraps a copy of the given array vals."""
        return vals.view(cls)

    @classmethod
    def from_list(cls, vals: List[float]) -> DataVector:
        """Create a DataVector from the given list of floats."""
        array = np.array(vals)
        return cls.create(array)


class TheoryVector(npt.NDArray[np.float64]):
    """This class wraps a np.ndarray that represents an observation predicted by
    some theory."""

    @classmethod
    def create(cls, vals: npt.NDArray[np.float64]) -> TheoryVector:
        """Create a TheoryVector that wraps a copy of the given array vals."""
        return vals.view(cls)

    @classmethod
    def from_list(cls, vals: List[float]) -> TheoryVector:
        """Create a TheoryVector from the given list of floats."""
        array = np.array(vals)
        return cls.create(array)


def residuals(data: DataVector, theory: TheoryVector) -> npt.NDArray[np.float64]:
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

    def residuals(self) -> npt.NDArray[np.float64]:
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


class StatisticUnreadError(RuntimeError):
    def __init__(self, stat: Statistic):
        msg = (
            f"The statistic {stat} was used for calculation before `read` "
            f"was called"
        )
        super().__init__(msg)
        self.statstic = stat


class Statistic(Updatable):
    """An abstract statistic class.

    Statistics read data from a SACC object as part of a multi-phase
    initialization. The manage a :python:`DataVector` and, given a
    :python:`ModelingTools` object, can compute a :python:`TheoryVector`.

    Statistics represent things like two-point functions and mass functions.
    """

    systematics: List[SourceSystematic]
    sacc_indices: npt.NDArray[np.int64]

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file."""

    @abstractmethod
    def get_data_vector(self) -> DataVector:
        """Gets the statistic data vector."""

    @abstractmethod
    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, applying any systematics."""
