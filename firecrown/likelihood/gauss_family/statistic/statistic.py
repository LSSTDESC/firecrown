"""

Gaussian Family Statistic Module
================================

The Statistic class describing objects that implement methods to compute the
data and theory vectors for a GaussianFamily subclass.

"""

from __future__ import annotations
from typing import List, Optional, final
from dataclasses import dataclass
from abc import abstractmethod
import warnings
import numpy as np
import numpy.typing as npt
import sacc

import firecrown.parameters
from firecrown.parameters import DerivedParameterCollection, RequiredParameters
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import Updatable


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
    """Run-time error indicating an attempt has been made to use a statistic
    that has not had `read` called in it."""

    def __init__(self, stat: Statistic):
        msg = (
            f"The statistic {stat} was used for calculation before `read` "
            f"was called.\nIt may be that a likelihood factory function did not"
            f"call `read` before returning the likelihood."
        )
        super().__init__(msg)
        self.statstic = stat


class Statistic(Updatable):
    """The abstract base class for all physics-related statistics.

    Statistics read data from a SACC object as part of a multi-phase
    initialization. They manage a :class:`DataVector` and, given a
    :class:`ModelingTools` object, can compute a :class:`TheoryVector`.

    Statistics represent things like two-point functions and mass functions.
    """

    def __init__(self, parameter_prefix: Optional[str] = None):
        super().__init__(parameter_prefix=parameter_prefix)
        self.sacc_indices: Optional[npt.NDArray[np.int64]]
        self.ready = False
        self.computed_theory_vector = False
        # TODO: how to we call this?
        self.predicted_statistic_: Optional[TheoryVector] = None

    def read(self, _: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC data, and mark it
        as ready for use. Derived classes that override this function
        should make sure to call the base class method using:
            super().read(sacc_data)
        as the last thing they do in `__init__`.

        Note that currently the argument is not used; it is present so that this
        method will have the correct argument type for the override.
        """
        self.ready = True
        if len(self.get_data_vector()) == 0:
            raise RuntimeError(
                f"the statistic {self} has read a data vector "
                f"of length 0; the length must be positive"
            )

    @abstractmethod
    def get_data_vector(self) -> DataVector:
        """Gets the statistic data vector."""

    @abstractmethod
    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, applying any systematics."""

    def get_theory_vector(self) -> TheoryVector:
        if not self.computed_theory_vector:
            raise RuntimeError(
                f"The theory for statistic {self} has not been computed yet."
            )
        # TODO: give proper error message
        assert self.predicted_statistic_ is not None
        return self.predicted_statistic_


class GuardedStatistic(Updatable):
    """:class:`GuardedStatistic` is used by the framework to maintain and
    validate the state of instances of classes derived from
    :class:`Statistic`."""

    def __init__(self, stat: Statistic):
        """Initialize the GuardedStatistic to contain the given
        :class:`Statistic`."""
        super().__init__()
        self.statistic = stat

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read whatever data is needed from the given :class:`sacc.Sacc
        object.

        After this function is called, the object should be prepared for the
        calling of the methods :meth:`get_data_vector` and
        :meth:`compute_theory_vector`.
        """
        if self.statistic.ready:
            raise RuntimeError("Firecrown has called read twice on a GuardedStatistic")
        try:
            self.statistic.read(sacc_data)
        except TypeError as exc:
            msg = (
                f"A statistic of type {type(self.statistic).__name__} has raised "
                f"an exception during `read`.\nThe problem may be a malformed "
                f"SACC data object."
            )
            raise RuntimeError(msg) from exc

    def get_data_vector(self) -> DataVector:
        """Return the contained :class:`Statistic`'s data vector.

        :class:`GuardedStatistic` ensures that :meth:`read` has been called.
        first."""
        if not self.statistic.ready:
            raise StatisticUnreadError(self.statistic)
        return self.statistic.get_data_vector()

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Return the contained :class:`Statistic`'s computed theory vector.

        :class:`GuardedStatistic` ensures that :meth:`read` has been called.
        first."""
        if not self.statistic.ready:
            raise StatisticUnreadError(self.statistic)
        return self.statistic.compute_theory_vector(tools)


class TrivialStatistic(Statistic):
    """A minimal statistic only to be used for testing Gaussian likelihoods.

    It returns a :class:`DataVector` and :class:`TheoryVector` each of which is
    three elements long. The SACC data provided to :meth:`TrivialStatistic.read`
    must supply the necessary values.
    """

    def __init__(self) -> None:
        """Initialize this statistic."""
        super().__init__()
        # Data and theory will both be of length self.count
        self.count = 3
        self.data_vector: Optional[DataVector] = None
        self.mean = firecrown.parameters.register_new_updatable_parameter()
        self.computed_theory_vector = False

    def read(self, sacc_data: sacc.Sacc):
        """Read the necessary items from the sacc data."""

        our_data = sacc_data.get_mean(data_type="count")
        assert len(our_data) == self.count
        self.data_vector = DataVector.from_list(our_data)
        self.sacc_indices = np.arange(len(self.data_vector))
        super().read(sacc_data)

    @final
    def _reset(self):
        """Reset this statistic."""
        self.computed_theory_vector = False

    @final
    def _required_parameters(self) -> RequiredParameters:
        """Return an empty RequiredParameters."""
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Return an empty DerivedParameterCollection."""
        return DerivedParameterCollection([])

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none."""
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, _: ModelingTools) -> TheoryVector:
        """Return a fixed theory vector."""
        self.computed_theory_vector = True
        self.predicted_statistic_ = TheoryVector.from_list([self.mean] * self.count)
        return self.predicted_statistic_
