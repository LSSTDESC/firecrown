"""The Student-t likelihood.

"""

from __future__ import annotations
from typing import List, final

import numpy as np

import pyccl

from .gauss_family import GaussFamily
from .statistic.statistic import Statistic
from ...parameters import ParamsMap, RequiredParameters, DerivedParameterCollection


class StudentT(GaussFamily):
    """A T-distribution for the log-likelihood.

    This distribution is appropriate when the covariance has been obtained
    from a finite number of simulations. See Sellentin & Heavens
    (2016; arXiv:1511.05969). As the number of simulations increases, the
    T-distribution approaches a Gaussian.

    :param statistics: List of statistics to build the theory and data vectors
    :param nu: The Student-t $\\nu$ parameter
    """

    def __init__(self, statistics: List[Statistic], nu: float):
        super().__init__(statistics)
        self.nu = nu  # pylint: disable-msg=C0103

    def compute_loglike(self, cosmo: pyccl.Cosmology):
        """Compute the log-likelihood.

        :param cosmo: Current Cosmology object
        """

        chi2 = self.compute_chisq(cosmo)
        return -0.5 * self.nu * np.log(1.0 + chi2 / (self.nu - 1.0))

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
