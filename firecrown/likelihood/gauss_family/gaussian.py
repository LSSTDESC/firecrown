"""

Gaussian Likelihood Module
==========================

Some notes.

"""


from __future__ import annotations
from typing import final
import pyccl

from .gauss_family import GaussFamily
from ...parameters import ParamsMap, RequiredParameters, DerivedParameterCollection


class ConstGaussian(GaussFamily):
    """A Gaussian log-likelihood with a constant covariance matrix."""

    def compute_loglike(self, cosmo: pyccl.Cosmology):
        """Compute the log-likelihood."""

        return -0.5 * self.compute_chisq(cosmo)

    @final
    def _update_gaussian_family(self, params: ParamsMap):
        pass

    @final
    def _reset_gaussian_family(self):
        pass

    @final
    def required_parameters_gaussian_family(self):
        return RequiredParameters([])

    @final
    def _get_derived_parameters_gaussian_family(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])
