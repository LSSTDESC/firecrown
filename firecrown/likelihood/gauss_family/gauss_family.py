"""

Gaussian Family Module
======================

Some notes.

"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, Sequence, Dict
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

        for i, s in enumerate(statistics):
            if not isinstance(s, Statistic):
                raise ValueError(
                    f"statistics[{i}] is not an instance of Statistic: {s}"
                    f"it is a {type(s)} instead."
                )

        self.statistics: UpdatableCollection[GuardedStatistic] = UpdatableCollection(
            GuardedStatistic(s) for s in statistics
        )
        self.cov: Optional[npt.NDArray[np.float64]] = None
        self.cholesky: Optional[npt.NDArray[np.float64]] = None
        self.inv_cov: Optional[npt.NDArray[np.float64]] = None
        self.cov_index_map: Optional[Dict[int, int]] = None
        self.computed_theory_vector = False
        self.theory_vector: Optional[npt.NDArray[np.double]] = None
        self.data_vector: Optional[npt.NDArray[np.double]] = None

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

        self.computed_theory_vector = False
        self.theory_vector = None

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

        indices_list = []
        data_vector_list = []
        for stat in self.statistics:
            stat.read(sacc_data)
            if stat.statistic.sacc_indices is None:
                raise RuntimeError(
                    f"The statistic {stat.statistic} has no sacc_indices."
                )
            indices_list.append(stat.statistic.sacc_indices.copy())
            data_vector_list.append(stat.statistic.get_data_vector())

        indices = np.concatenate(indices_list)
        data_vector = np.concatenate(data_vector_list)
        cov = np.zeros((len(indices), len(indices)))

        for new_i, old_i in enumerate(indices):
            for new_j, old_j in enumerate(indices):
                cov[new_i, new_j] = covariance[old_i, old_j]

        self.data_vector = data_vector
        self.cov_index_map = {old_i: new_i for new_i, old_i in enumerate(indices)}
        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)
        self.inv_cov = np.linalg.inv(cov)

        self.state = State.READY

    def write(self, sacc_data: sacc.Sacc, strict=True) -> sacc.Sacc:
        new_sacc = sacc_data.copy()

        sacc_indices_list = []
        predictions_list = []
        for stat in self.statistics:
            assert stat.statistic.sacc_indices is not None
            sacc_indices_list.append(stat.statistic.sacc_indices.copy())
            predictions_list.append(stat.statistic.get_theory_vector())

        sacc_indices = np.concatenate(sacc_indices_list)
        predictions = np.concatenate(predictions_list)
        assert len(sacc_indices) == len(predictions)

        if strict:
            if set(sacc_indices.tolist()) != set(sacc_data.indices()):
                raise RuntimeError(
                    "The predicted data does not cover all the data in the "
                    "sacc object. To write only the calculated predictions, "
                    "set strict=False."
                )

        for prediction_idx, sacc_idx in enumerate(sacc_indices):
            new_sacc.data[sacc_idx].value = predictions[prediction_idx]

        return new_sacc

    @final
    def get_cov(self, statistic: Optional[Statistic] = None) -> npt.NDArray[np.float64]:
        """Gets the current covariance matrix.

        :param statistic: The statistic for which the sub-covariance matrix
        should be return. If not specified, return the covariance of all
        statistics.
        """
        assert self._is_ready(), "read() must be called before get_cov()"
        assert self.cov is not None
        if statistic is not None:
            assert statistic.sacc_indices is not None
            assert self.cov_index_map is not None
            idx = [self.cov_index_map[idx] for idx in statistic.sacc_indices]
            # We do not change the state.
            return self.cov[np.ix_(idx, idx)]
        # We do not change the state.
        return self.cov

    @final
    def get_data_vector(self) -> npt.NDArray[np.float64]:
        """Get the data vector from all statistics and concatenate in the right
        order."""
        assert self._is_ready(), "read() must be called before get_data_vector()"

        assert self.data_vector is not None
        return self.data_vector

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
        self.computed_theory_vector = True
        self.theory_vector = np.concatenate(theory_vector_list)

        return self.theory_vector

    @final
    def get_theory_vector(self) -> npt.NDArray[np.float64]:
        """Get the theory vector from all statistics and concatenate in the right
        order."""

        assert (
            self.state == State.UPDATED
        ), "update() must be called before get_theory_vector()"

        if not self.computed_theory_vector:
            raise RuntimeError(
                "The theory vector has not been computed yet. "
                "Call compute_theory_vector first."
            )
        assert self.theory_vector is not None, (
            "Implementation error, "
            "computed_theory_vector is True but theory_vector is None"
        )
        return self.theory_vector

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

        x = scipy.linalg.solve_triangular(self.cholesky, residuals, lower=True)
        chisq = np.dot(x, x)

        # We do not change the state.
        return chisq

    def _is_ready(self) -> bool:
        """Return True if the state is either READY or UPDATED."""
        return self.state in (State.READY, State.UPDATED)
