"""

Gaussian Likelihood Module
==========================

Some notes.

"""

from __future__ import annotations
from typing import final

from .gauss_family import GaussFamily
from ...parameters import ParamsMap, RequiredParameters, DerivedParameterCollection
from ...modeling_tools import ModelingTools


class ConstGaussian(GaussFamily):
    """A Gaussian log-likelihood with a constant covariance matrix."""

    def compute_loglike(self, tools: ModelingTools):
        """Compute the log-likelihood."""

        return -0.5 * self.compute_chisq(tools)

    @final
    def _reset_gaussian_family(self):
        pass

    @final
    def _required_parameters_gaussian_family(self):
        return RequiredParameters([])

    @final
    def _get_derived_parameters_gaussian_family(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])
