"""

Gaussian Family Module
======================

Some notes.

"""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence, Dict
from typing import final
import warnings

import numpy as np
import numpy.typing as npt
import scipy.linalg

import sacc

from ..likelihood import Likelihood
from ...modeling_tools import ModelingTools
from ...updatable import UpdatableCollection
from .statistic.statistic import Statistic, GuardedStatistic


class GaussFamily(Likelihood):
    """GaussFamily is an abstract class. It is the base class for all likelihoods
    based on a chi-squared calculation. It provides an implementation of
    Likelihood.compute_chisq. Derived classes must implement the abstract method
    compute_loglike, which is inherited from Likelihood.
    """

    def __init__(
        self,
        statistics: Sequence[Statistic],
    ):
        super().__init__()
        if len(statistics) == 0:
            raise ValueError("GaussFamily requires at least one statistic")
        self.statistics: UpdatableCollection[GuardedStatistic] = UpdatableCollection(
            GuardedStatistic(s) for s in statistics
        )
        self.cov: Optional[npt.NDArray[np.float64]] = None
        self.cholesky: Optional[npt.NDArray[np.float64]] = None
        self.inv_cov: Optional[npt.NDArray[np.float64]] = None
        self.cov_index_map: Optional[Dict[int, int]] = None
        self.predicted_data_vector: Optional[npt.NDArray[np.double]] = None
        self.measured_data_vector: Optional[npt.NDArray[np.double]] = None

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrix for this likelihood from the SACC file."""

        if sacc_data.covariance is None:
            msg = (
                f"The {type(self).__name__} likelihood requires a covariance, "
                f"but the SACC data object being read does not have one."
            )
            raise RuntimeError(msg)

        covariance = sacc_data.covariance.dense
        for stat in self.statistics:
            stat.read(sacc_data)

        indices_list = [
            s.statistic.sacc_indices.copy()
            for s in self.statistics
            if s.statistic.sacc_indices is not None
        ]
        indices = np.concatenate(indices_list)
        cov = np.zeros((len(indices), len(indices)))

        for new_i, old_i in enumerate(indices):
            for new_j, old_j in enumerate(indices):
                cov[new_i, new_j] = covariance[old_i, old_j]

        self.cov_index_map = {old_i: new_i for new_i, old_i in enumerate(indices)}
        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)
        self.inv_cov = np.linalg.inv(cov)

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
        assert self.cov is not None
        if statistic is not None:
            assert statistic.sacc_indices is not None
            assert self.cov_index_map is not None
            idx = [self.cov_index_map[idx] for idx in statistic.sacc_indices]
            return self.cov[np.ix_(idx, idx)]
        return self.cov

    @final
    def get_data_vector(self) -> npt.NDArray[np.float64]:
        """Get the data vector from all statistics and concatenate in the right
        order."""

        data_vector_list: List[npt.NDArray[np.float64]] = [
            stat.get_data_vector() for stat in self.statistics
        ]
        return np.concatenate(data_vector_list)

    @final
    def compute_theory_vector(self, tools: ModelingTools) -> npt.NDArray[np.float64]:
        """Computes the theory vector using the current instance of pyccl.Cosmology.

        :param tools: Current ModelingTools object
        """

        theory_vector_list: List[npt.NDArray[np.float64]] = [
            stat.compute_theory_vector(tools) for stat in self.statistics
        ]
        return np.concatenate(theory_vector_list)

    @final
    def compute(
        self, tools: ModelingTools
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate and return both the data and theory vectors."""

        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "The use of the `compute` method on Statistic is deprecated."
            "The Statistic objects should implement `get_data` and "
            "`compute_theory_vector` instead.",
            category=DeprecationWarning,
        )

        return self.get_data_vector(), self.compute_theory_vector(tools)

    @final
    def compute_chisq(self, tools: ModelingTools) -> float:
        """Calculate and return the chi-squared for the given cosmology."""
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

        self.predicted_data_vector = theory_vector
        self.measured_data_vector = data_vector

        x = scipy.linalg.solve_triangular(self.cholesky, residuals, lower=True)
        chisq = np.dot(x, x)

        return chisq
