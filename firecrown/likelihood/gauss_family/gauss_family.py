from __future__ import annotations
from typing import List, Optional
from abc import ABC, abstractmethod
from typing import final
import numpy as np
import scipy.linalg

from ..likelihood import Likelihood
from .statistic.statistic import Statistic

class GaussFamily(Likelihood):
    """A Gaussian log-likelihood with a constant covariance matrix.

    Parameters
    ----------
    statistics : list of Statistic
        A list of the statistics 

    Attributes
    ----------
    cov : np.ndarray, shape (n, n)
        The covariance matrix.
    cholesky : np.ndarray, shape (n, n)
        The (lower triangular) Cholesky decomposition of the covariance matrix.
    inv_cov : np.ndarray, shape (n, n)
        The inverse of the covariance matrix.

    Methods
    -------
    compute_loglike : compute the log-likelihood
    """

    def __init__(self, statistics: List[Statistic]):
        self.statistics = statistics
        self.cov: Optional[np.ndarray] = None
        self.cholesky: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrirx for this likelihood from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        _sd = sacc_data.copy()
        inds = []
        for stat in self.statistics:
            stat.read (sacc_data)
            inds.append(stat.sacc_inds.copy())
            
        inds = np.concatenate(inds, axis=0)
        cov = np.zeros((len(inds), len(inds)))
        for new_i, old_i in enumerate(inds):
            for new_j, old_j in enumerate(inds):
                cov[new_i, new_j] = _sd.covariance.dense[old_i, old_j]
        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)
        self.inv_cov = np.linalg.inv(cov)

    @final
    def compute_chisq(self, cosmo: pyccl.Cosmology, params: Dict[str, float]):
        """Compute the log-likelihood.

        Parameters
        ----------

        Returns
        -------
        loglike : float
            The log-likelihood.
        """

        dv = []
        for stat in self.statistics:
            stat.update_params(params)
            data, theory = stat.compute(cosmo, params)

            dv.append(np.atleast_1d(data - theory))

        dv = np.concatenate(dv, axis=0)
        x = scipy.linalg.solve_triangular(self.cholesky, dv, lower=True)
        return np.dot(x, x)

