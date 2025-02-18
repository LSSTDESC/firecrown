"""The Student-t likelihood."""

from __future__ import annotations

import numpy as np


from firecrown.likelihood.gaussfamily import GaussFamily
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.statistic import Statistic
from firecrown import parameters


class StudentT(GaussFamily):
    r"""A T-distribution for the log-likelihood.

    This distribution is appropriate when the covariance has been obtained
    from a finite number of simulations. See Sellentin & Heavens
    (2016; arXiv:1511.05969). As the number of simulations increases, the
    T-distribution approaches a Gaussian.
    """

    def __init__(
        self,
        statistics: list[Statistic],
        nu: None | float = None,
    ):
        """Initialize a StudentT object.

        :param statistics: The statistics to use in the likelihood
        :param nu: The degrees of freedom of the T-distribution
        """
        super().__init__(statistics)
        self.nu = parameters.register_new_updatable_parameter(nu, default_value=3.0)

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Compute the log-likelihood.

        :param tools: The modeling tools used to compute the likelihood.
        :return: The log-likelihood.
        """
        chi2 = self.compute_chisq(tools)
        return -0.5 * self.nu * np.log(1.0 + chi2 / (self.nu - 1.0))
