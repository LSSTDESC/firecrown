"""

Gaussian Family Module
======================

Some notes.

"""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence
from typing import final
import warnings

import numpy as np
import numpy.typing as npt
import scipy.linalg

import sacc

from ..likelihood import Likelihood
from ...modeling_tools import ModelingTools
from ...updatable import UpdatableCollection
from .statistic.statistic import Statistic


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
        self.statistics = UpdatableCollection(statistics)
        self.cov: Optional[npt.NDArray[np.float64]] = None
        self.cholesky: Optional[npt.NDArray[np.float64]] = None
        self.inv_cov: Optional[npt.NDArray[np.float64]] = None

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrix for this likelihood from the SACC file."""

        covariance = sacc_data.covariance.dense
        for stat in self.statistics:
            stat.read(sacc_data)

        indices_list = [stat.sacc_indices.copy() for stat in self.statistics]
        indices = np.concatenate(indices_list)
        cov = np.zeros((len(indices), len(indices)))

        for new_i, old_i in enumerate(indices):
            for new_j, old_j in enumerate(indices):
                cov[new_i, new_j] = covariance[old_i, old_j]

        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)
        self.inv_cov = np.linalg.inv(cov)

    @final
    def get_cov(self) -> npt.NDArray[np.float64]:
        """Gets the current covariance matrix."""
        assert self.cov is not None
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
        residuals = data_vector - theory_vector

        self.predicted_data_vector: npt.NDArray[np.float64] = theory_vector
        self.measured_data_vector: npt.NDArray[np.float64] = data_vector

        x = scipy.linalg.solve_triangular(self.cholesky, residuals, lower=True)
        chisq = np.dot(x, x)

        return chisq
