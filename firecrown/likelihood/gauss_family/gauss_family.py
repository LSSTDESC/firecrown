"""

Gaussian Family Module
======================

Some notes.

"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Tuple, Sequence
from typing import final
import warnings

import numpy as np
import numpy.typing as npt
import scipy.linalg

import sacc

from ...parameters import ParamsMap
from ..likelihood import Likelihood
from ...modeling_tools import ModelingTools
from ...updatable import UpdatableCollection
from .statistic.statistic import Statistic, GuardedStatistic


class State(Enum):
    """The states used in GaussFamily."""

    INITIALIZED = 1
    READY = 2
    UPDATED = 3


class GaussFamily(Likelihood):
    """GaussFamily is an abstract class. It is the base class for all likelihoods
    based on a chi-squared calculation. It provides an implementation of
    Likelihood.compute_chisq. Derived classes must implement the abstract method
    compute_loglike, which is inherited from Likelihood.

    GaussFamily (and all classes that inherit from it) must abide by the the
    following rules regarding the order of calling of methods.

      1. after a new object is created, :meth:`read` must be called before any
         other method in the interfaqce.
      2. after :meth:`read` has been called it is legal to call
         :meth:`get_data_vector`, or to call :meth:`update`.
      3. after :meth:`update` is called it is then legal to call
         :meth:`calculate_loglike` or :meth:`get_data_vector`, or to reset
         the object (returning to the pre-update state) by calling
         :meth:`reset`.
    """

    def __init__(
        self,
        statistics: Sequence[Statistic],
    ):
        super().__init__()
        self.state: State = State.INITIALIZED
        if len(statistics) == 0:
            raise ValueError("GaussFamily requires at least one statistic")
        self.statistics: UpdatableCollection = UpdatableCollection(
            GuardedStatistic(s) for s in statistics
        )
        self.cov: Optional[npt.NDArray[np.float64]] = None
        self.cholesky: Optional[npt.NDArray[np.float64]] = None
        self.inv_cov: Optional[npt.NDArray[np.float64]] = None

    def _update(self, _: ParamsMap) -> None:
        """Handle the state resetting required by :class:`GaussFamily`
        likelihoods. Any derived class that needs to implement :meth:`_update`
        for its own reasons must be sure to do what this does: check the state
        at the start of the method, and change the state at the end of the
        method."""
        assert self.state == State.READY, "read() must be called before update()"
        self.state = State.UPDATED

    def _reset(self) -> None:
        """Handle the state resetting required by :class:`GaussFamily`
        likelihoods. Any derived class that needs to implement :meth:`reset`
        for its own reasons must be sure to do what this does: check the state
        at the start of the method, and change the state at the end of the
        method."""
        assert self.state == State.UPDATED, "update() must be called before reset()"
        self.state = State.READY

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrix for this likelihood from the SACC file."""

        assert self.state == State.INITIALIZED, "read() must only be called once"
        if sacc_data.covariance is None:
            msg = (
                f"The {type(self).__name__} likelihood requires a covariance, "
                f"but the SACC data object being read does not have one."
            )
            raise RuntimeError(msg)

        covariance = sacc_data.covariance.dense
        for stat in self.statistics:
            stat.read(sacc_data)

        indices_list = [s.statistic.sacc_indices.copy() for s in self.statistics]
        indices = np.concatenate(indices_list)
        cov = np.zeros((len(indices), len(indices)))

        for new_i, old_i in enumerate(indices):
            for new_j, old_j in enumerate(indices):
                cov[new_i, new_j] = covariance[old_i, old_j]

        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)
        self.inv_cov = np.linalg.inv(cov)

        self.state = State.READY

    @final
    def get_cov(self) -> npt.NDArray[np.float64]:
        """Gets the current covariance matrix."""
        assert self._is_ready(), "read() must be called before get_cov()"
        assert self.cov is not None
        # We do not change the state.
        return self.cov

    @final
    def get_data_vector(self) -> npt.NDArray[np.float64]:
        """Get the data vector from all statistics and concatenate in the right
        order."""
        assert self._is_ready(), "read() must be called before get_data_vector()"

        data_vector_list: List[npt.NDArray[np.float64]] = [
            stat.get_data_vector() for stat in self.statistics
        ]
        # We do not change the state.
        return np.concatenate(data_vector_list)

    @final
    def compute_theory_vector(self, tools: ModelingTools) -> npt.NDArray[np.float64]:
        """Computes the theory vector using the current instance of pyccl.Cosmology.

        :param tools: Current ModelingTools object
        """
        assert (
            self.state == State.UPDATED
        ), "update() must be called before compute_theory_vector()"

        theory_vector_list: List[npt.NDArray[np.float64]] = [
            stat.compute_theory_vector(tools) for stat in self.statistics
        ]
        # We do not change the state
        return np.concatenate(theory_vector_list)

    @final
    def compute(
        self, tools: ModelingTools
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate and return both the data and theory vectors."""
        assert self.state == State.UPDATED, "update() must be called before compute()"
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "The use of the `compute` method on Statistic is deprecated."
            "The Statistic objects should implement `get_data` and "
            "`compute_theory_vector` instead.",
            category=DeprecationWarning,
        )

        # We do not change the state.
        return self.get_data_vector(), self.compute_theory_vector(tools)

    @final
    def compute_chisq(self, tools: ModelingTools) -> float:
        """Calculate and return the chi-squared for the given cosmology."""
        assert (
            self.state == State.UPDATED
        ), "update() must be called before compute_chisq()"
        theory_vector: npt.NDArray[np.float64]
        data_vector: npt.NDArray[np.float64]
        residuals: npt.NDArray[np.float64]
        try:
            theory_vector = self.compute_theory_vector(tools)
            data_vector = self.get_data_vector()
        except NotImplementedError:
            data_vector, theory_vector = self.compute(tools)

        assert len(data_vector) == len(theory_vector)
        residuals = data_vector - theory_vector

        self.predicted_data_vector: npt.NDArray[np.float64] = theory_vector
        self.measured_data_vector: npt.NDArray[np.float64] = data_vector

        x = scipy.linalg.solve_triangular(self.cholesky, residuals, lower=True)
        chisq = np.dot(x, x)

        # We do not change the state.
        return chisq

    def _is_ready(self) -> bool:
        """Return True if the state is either READY or UPDATED."""
        return self.state in (State.READY, State.UPDATED)
