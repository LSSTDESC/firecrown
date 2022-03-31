from __future__ import annotations
from typing import List, Optional
import numpy as np
import scipy.linalg

from .gauss_family import GaussFamily

class ConstGaussian(GaussFamily):
    """A Gaussian log-likelihood with a constant covariance matrix.

    Methods
    -------
    compute_loglike : compute the log-likelihood
    """

    def compute_loglike(self, cosmo: pyccl.Cosmology, params: Dict[str, float]):
        """Compute the log-likelihood.

        Parameters
        ----------

        Returns
        -------
        loglike : float
            The log-likelihood.
        """

        return -0.5 * self.compute_chisq(cosmo, params)
