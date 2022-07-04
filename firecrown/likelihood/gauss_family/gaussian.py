from __future__ import annotations
from typing import final
import pyccl

from .gauss_family import GaussFamily
from ...parameters import ParamsMap, RequiredParameters


class ConstGaussian(GaussFamily):
    """A Gaussian log-likelihood with a constant covariance matrix."""

    def compute_loglike(self, cosmo: pyccl.Cosmology):
        """Compute the log-likelihood."""

        return -0.5 * self.compute_chisq(cosmo)

    @final
    def _update_gaussian_family(self, params: ParamsMap):
        pass

    @final
    def required_parameters_gaussian_family(self):
        return RequiredParameters([])
