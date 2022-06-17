from __future__ import annotations
from typing import List, final

import numpy as np

import pyccl

from .gauss_family import GaussFamily
from .statistic.statistic import Statistic
from ...parameters import ParamsMap, RequiredParameters


class StudentT(GaussFamily):
    """A T-distribution for the log-likelihood.

    This distribution is appropriate when the covariance has been obtained
    from a finite number of simulations. See Sellentin & Heavens
    (2016; arXiv:1511.05969). As the number of simulations increases, the
    T-distribution approaches a Gaussian.

    Parameters
    ----------
    statistics : list of Statistic
        A list of the statistics
    nu: int
        The shape parameter. Set to the number of simulations.

    Methods
    -------
    compute_loglike : compute the log-likelihood
    """

    def __init__(self, statistics: List[Statistic], nu):
        super().__init__(statistics)
        self.nu = nu

    def compute_loglike(self, cosmo: pyccl.Cosmology):
        """Compute the log-likelihood.

        Parameters
        ----------

        Returns
        -------
        loglike : float
            The log-likelihood.
        """

        chi2 = self.compute_chisq(cosmo)
        return -0.5 * self.nu * np.log(1.0 + chi2 / (self.nu - 1.0))

    @final
    def _update_gaussian_family(self, params: ParamsMap):
        pass

    @final
    def required_parameters_gaussian_family(self):
        return RequiredParameters([])
