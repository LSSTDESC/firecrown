"""The Student-t likelihood.

"""

from __future__ import annotations
from typing import List, Optional, final

import numpy as np

from .gauss_family import GaussFamily
from ...modeling_tools import ModelingTools
from .statistic.statistic import Statistic
from ... import parameters
from ...parameters import RequiredParameters, DerivedParameterCollection


class StudentT(GaussFamily):
    """A T-distribution for the log-likelihood.

    This distribution is appropriate when the covariance has been obtained
    from a finite number of simulations. See Sellentin & Heavens
    (2016; arXiv:1511.05969). As the number of simulations increases, the
    T-distribution approaches a Gaussian.

    :param statistics: List of statistics to build the theory and data vectors
    :param nu: The Student-t $\\nu$ parameter
    """

    def __init__(
        self,
        statistics: List[Statistic],
        nu: Optional[float],
    ):
        super().__init__(statistics)
        self.nu = parameters.create(nu)

    def compute_loglike(self, tools: ModelingTools):
        """Compute the log-likelihood.

        :param cosmo: Current Cosmology object
        """

        ccl_cosmo = tools.get_ccl_cosmology()
        chi2 = self.compute_chisq(ccl_cosmo)
        return -0.5 * self.nu * np.log(1.0 + chi2 / (self.nu - 1.0))

    @final
    def _reset_gaussian_family(self):
        pass

    @final
    def _required_parameters_gaussian_family(self):
        return RequiredParameters([])

    @final
    def _get_derived_parameters_gaussian_family(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])
