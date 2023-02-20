"""

Gaussian Family Module
======================

Some notes.

"""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence
from typing import final
from abc import abstractmethod
import warnings

import numpy as np
import numpy.typing as npt
import scipy.linalg

import sacc

from ..likelihood import Likelihood
from ...modeling_tools import ModelingTools
from ...updatable import UpdatableCollection
from .statistic.statistic import Statistic
from ...parameters import ParamsMap, RequiredParameters, DerivedParameterCollection


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

        # pylint: disable-next=C0103
        x = scipy.linalg.solve_triangular(self.cholesky, residuals, lower=True)
        chisq = np.dot(x, x)

        return chisq

    @final
    def _update(self, params: ParamsMap) -> None:
        """Implementation of the Likelihood interface method _update.

        This updates all statistics and calls teh abstract method
        _update_gaussian_family."""
        self.statistics.update(params)
        self._update_gaussian_family(params)

    @final
    def _reset(self) -> None:
        """Implementation of Likelihood interface method _reset.

        This resets all statistics and calls the abstract method
        _reset_gaussian_family."""
        self._reset_gaussian_family()
        self.statistics.reset()

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = (
            self._get_derived_parameters_gaussian_family()
            + self.statistics.get_derived_parameters()
        )

        return derived_parameters

    @abstractmethod
    def _update_gaussian_family(self, params: ParamsMap) -> None:
        """Abstract method to update GaussianFamily state. Must be implemented by all
        subclasses."""

    @abstractmethod
    def _reset_gaussian_family(self) -> None:
        """Abstract method to reset GaussianFamily state. Must be implemented by all
        subclasses."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        """Return a RequiredParameters object containing the information for
        this Updatable.

        This includes the required parameters for all statistics, as well as those
        for the derived class.

        Derived classes must implement required_parameters_gaussian_family."""
        stats_rp = self.statistics.required_parameters()
        stats_rp = self._required_parameters_gaussian_family() + stats_rp

        return stats_rp

    @abstractmethod
    def _required_parameters_gaussian_family(self):
        """Required parameters for GaussFamily subclasses."""

    @abstractmethod
    def _get_derived_parameters_gaussian_family(self) -> DerivedParameterCollection:
        """Get derived parameters for GaussFamily subclasses."""
