"""The Student-t likelihood."""

from __future__ import annotations
from typing import Optional

import numpy as np

from .gauss_family import GaussFamily
from ...modeling_tools import ModelingTools
from .statistic.statistic import Statistic
from ... import parameters


class StudentT(GaussFamily):
    r"""A T-distribution for the log-likelihood.

    This distribution is appropriate when the covariance has been obtained
    from a finite number of simulations. See Sellentin & Heavens
    (2016; arXiv:1511.05969). As the number of simulations increases, the
    T-distribution approaches a Gaussian.

    :param statistics: list of statistics to build the theory and data vectors
    :param nu: The Student-t $\nu$ parameter
    """

    def __init__(
        self,
        statistics: list[Statistic],
        nu: Optional[float] = None,
    ):
        super().__init__(statistics)
        self.nu = parameters.register_new_updatable_parameter(nu)

    def compute_loglike(self, tools: ModelingTools):
        """Compute the log-likelihood.

        :param cosmo: Current Cosmology object
        """
        chi2 = self.compute_chisq(tools)
        return -0.5 * self.nu * np.log(1.0 + chi2 / (self.nu - 1.0))
