"""

Gaussian Family Module
======================

Some notes.

"""

from __future__ import annotations
from typing import List, Optional
from typing import final
from abc import abstractmethod

import numpy as np
import scipy.linalg

import pyccl
import sacc

from ..likelihood import Likelihood
from ...updatable import UpdatableCollection
from .statistic.statistic import Statistic
from ...parameters import ParamsMap, RequiredParameters, DerivedParameterCollection


class GaussFamily(Likelihood):
    """GaussFamily is an abstract class. It is the base class for all likelihoods
    based on a chi-squared calculation. It provides an implementation of
    Likelihood.compute_chisq. Derived classes must implement the abstract method
    compute_loglike, which is inherited from Likelihood.
    """

    def __init__(self, statistics: List[Statistic]):
        super().__init__()
        self.statistics = UpdatableCollection(statistics)
        self.cov: Optional[np.ndarray] = None
        self.cholesky: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrix for this likelihood from the SACC file."""

        _sd = sacc_data.copy()
        indices_list = []
        for stat in self.statistics:
            stat.read(sacc_data)
            indices_list.append(stat.sacc_indices.copy())

        indices = np.concatenate(indices_list, axis=0)
        cov = np.zeros((len(indices), len(indices)))
        for new_i, old_i in enumerate(indices):
            for new_j, old_j in enumerate(indices):
                cov[new_i, new_j] = _sd.covariance.dense[old_i, old_j]
        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)
        self.inv_cov = np.linalg.inv(cov)

    @final
    def get_cov(self) -> np.ndarray:
        """Gets the current covariance matrix."""
        assert self.cov is not None
        return self.cov

    @final
    def get_data_vector(self) -> np.ndarray:
        """Get the data vector from all statistics
        and concatenate in the right order."""
        data_vector_list: List[np.ndarray] = []
        for stat in self.statistics:
            data = stat.get_data_vector()
            data_vector_list.append(np.atleast_1d(data))
        return np.concatenate(data_vector_list)

    @final
    def compute_theory_vector(self, cosmo: pyccl.Cosmology) -> np.ndarray:
        theory_vector_list: List[np.ndarray] = []
        for stat in self.statistics:
            theory = stat.compute_theory_vector(cosmo)
            theory_vector_list.append(np.atleast_1d(theory))

        return np.concatenate(theory_vector_list)

    @final
    def compute_chisq(self, cosmo: pyccl.Cosmology) -> float:
        """Calculate and return the chi-squared for the given cosmology."""
        theory_vector: np.ndarray = self.compute_theory_vector(cosmo)
        data_vector: np.ndarray = self.get_data_vector()
        residuals: np.ndarray = data_vector - theory_vector

        self.predicted_data_vector: np.ndarray = theory_vector
        self.measured_data_vector: np.ndarray = data_vector

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
