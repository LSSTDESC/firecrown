"""Provides GaussFamily concrete types.

"""

from __future__ import annotations
import numpy as np

from .gauss_family import GaussFamily
from ...modeling_tools import ModelingTools


class ConstGaussian(GaussFamily):
    """A Gaussian log-likelihood with a constant covariance matrix."""

    def compute_loglike(self, tools: ModelingTools):
        """Compute the log-likelihood."""

        return -0.5 * self.compute_chisq(tools)

    def make_realization_vector(self) -> np.ndarray:
        theory_vector = self.get_theory_vector()
        assert self.cholesky is not None
        new_data_vector = theory_vector + np.dot(
            self.cholesky, np.random.randn(len(theory_vector))
        )

        return new_data_vector
