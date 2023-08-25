"""Provides GaussFamily concrete types.

"""

from __future__ import annotations

from .gauss_family import GaussFamily
from ...modeling_tools import ModelingTools


class ConstGaussian(GaussFamily):
    """A Gaussian log-likelihood with a constant covariance matrix."""

    def compute_loglike(self, tools: ModelingTools):
        """Compute the log-likelihood."""

        return -0.5 * self.compute_chisq(tools)
