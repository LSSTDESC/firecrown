"""Provides GaussFamily concrete types."""

from __future__ import annotations

import numpy as np

from firecrown.likelihood.gaussfamily import GaussFamily
from firecrown.modeling_tools import ModelingTools


class ConstGaussian(GaussFamily):
    """Base class for constant covariance Gaussian likelihoods.

    Provides shared implementations of compute_loglike and make_realization_vector
    for all constant covariance Gaussian likelihood variants.
    """

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Compute the log-likelihood.

        :params tools: The modeling tools used to compute the likelihood.
        :return: The log-likelihood.
        """
        return -0.5 * self.compute_chisq(tools)

    def make_realization_vector(self) -> np.ndarray:
        """Create a new (randomized) realization of the model.

        :return: A new realization of the model
        """
        theory_vector = self.get_theory_vector()
        assert self.cholesky is not None
        new_data_vector = theory_vector + np.dot(
            self.cholesky, np.random.randn(len(theory_vector))
        )

        return new_data_vector
