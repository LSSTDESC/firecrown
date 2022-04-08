from __future__ import annotations
import pyccl

from .gauss_family import GaussFamily
from firecrown.parameters import ParamsMap


class ConstGaussian(GaussFamily):
    """A Gaussian log-likelihood with a constant covariance matrix.

    Methods
    -------
    compute_loglike : compute the log-likelihood
    """

    def compute_loglike(self, cosmo: pyccl.Cosmology, params: ParamsMap):
        """Compute the log-likelihood.

        Parameters
        ----------

        Returns
        -------
        loglike : float
            The log-likelihood.
        """

        return -0.5 * self.compute_chisq(cosmo, params)
