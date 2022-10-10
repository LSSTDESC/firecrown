"""

Gaussian Family Module
======================

Some notes.

"""

from __future__ import annotations
from typing import List, Optional
from typing import final
import numpy as np
import scipy.linalg

import pyccl
import sacc

from ..likelihood import Likelihood
from ...updatable import UpdatableCollection
from .statistic.statistic import Statistic
from ...parameters import ParamsMap, RequiredParameters, DerivedParameterCollection
from abc import abstractmethod


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
        """Read the covariance matrirx for this likelihood from the SACC file."""

        _sd = sacc_data.copy()
        inds_list = []
        for stat in self.statistics:
            stat.read(sacc_data)
            inds_list.append(stat.sacc_inds.copy())

        inds = np.concatenate(inds_list, axis=0)
        cov = np.zeros((len(inds), len(inds)))
        for new_i, old_i in enumerate(inds):
            for new_j, old_j in enumerate(inds):
                cov[new_i, new_j] = _sd.covariance.dense[old_i, old_j]
        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)
        self.inv_cov = np.linalg.inv(cov)

    @final
    def compute_chisq(self, cosmo: pyccl.Cosmology) -> float:
        """Calculate and return the chi-squared for the given cosmology."""

        r = []
        theory_vector = []
        data_vector = []
        for stat in self.statistics:
            data, theory = stat.compute(cosmo)
            r.append(np.atleast_1d(data - theory))
            theory_vector.append(np.atleast_1d(theory))
            data_vector.append(np.atleast_1d(data))

        r = np.concatenate(r, axis=0)
        self.predicted_data_vector = np.concatenate(theory_vector)
        self.measured_data_vector = np.concatenate(data_vector)
        x = scipy.linalg.solve_triangular(self.cholesky, r, lower=True)
        chisq = np.dot(x, x)
        assert np.isscalar(chisq)
        return float(chisq)

    @final
    def _update(self, params: ParamsMap):
        self.statistics.update(params)
        self._update_gaussian_family(params)

    @final
    def _reset(self):
        self._reset_gaussian_family()
        self.statistics.reset()

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = self._get_derived_parameters_gaussian_family()

        (
            has_derived_parameters,
            derived_parameters0,
        ) = self.statistics.get_derived_parameters()
        if has_derived_parameters:
            derived_parameters = derived_parameters + derived_parameters0

        return derived_parameters

    @abstractmethod
    def _update_gaussian_family(self, params: ParamsMap):
        pass

    @abstractmethod
    def _reset_gaussian_family(self):
        pass

    @final
    def required_parameters(self) -> RequiredParameters:
        stats_rp = self.statistics.required_parameters()
        stats_rp = self.required_parameters_gaussian_family() + stats_rp

        return stats_rp

    @abstractmethod
    def required_parameters_gaussian_family(self):
        """Required parameters for GaussFamily subclasses."""

    @abstractmethod
    def _get_derived_parameters_gaussian_family(self) -> DerivedParameterCollection:
        """Get derived parameters for GaussFamily subclasses."""
