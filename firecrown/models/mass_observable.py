"""Cluster Mass Richness proxy module

Define the Cluster Mass Richness proxy module and its arguments.
"""
from typing import Tuple, List, Dict

import numpy as np
from scipy import special
from .. import parameters
from .kernel import Kernel, KernelType
import pdb


class MassRichnessMuSigma(Kernel):
    def __init__(
        self,
        pivot_mass,
        pivot_redshift,
        integral_bounds: List[Tuple[float, float]] = None,
    ):
        super().__init__(KernelType.mass_proxy, False, integral_bounds, True)
        self.pivot_mass = pivot_mass
        self.pivot_redshift = pivot_redshift
        self.pivot_mass = self.pivot_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        # Updatable parameters
        self.mu_p0 = parameters.create()
        self.mu_p1 = parameters.create()
        self.mu_p2 = parameters.create()
        self.sigma_p0 = parameters.create()
        self.sigma_p1 = parameters.create()
        self.sigma_p2 = parameters.create()
        self.limits = self.limits_generator()

        # Verify this gets called last or first

    def limits_generator(self):
        i = 0
        n = len(self.integral_bounds)

        while True:
            yield self.integral_bounds[i % n]
            i += 1

    def observed_value(self, p: Tuple[float, float, float], mass, z):
        """Return observed quantity corrected by redshift and mass."""

        ln_mass = mass * np.log(10)
        delta_ln_mass = ln_mass - self.pivot_mass
        delta_z = np.log1p(z) - self.log1p_pivot_redshift

        return p[0] + p[1] * delta_ln_mass + p[2] * delta_z

    def analytic_solution(
        self, args: List[float], index_lkp: Dict[str, int], args_lkp: Dict[str, int]
    ):
        mass = args[index_lkp["mass"]]
        z = args[index_lkp["z"]]
        observed_mean_mass = self.observed_value(
            (self.mu_p0, self.mu_p1, self.mu_p2),
            mass,
            z,
        )
        observed_mass_sigma = self.observed_value(
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            mass,
            z,
        )
        min_limit = args[2]
        max_limit = args[3]
        x_min = (observed_mean_mass - min_limit * np.log(10.0)) / (
            np.sqrt(2.0) * observed_mass_sigma
        )
        x_max = (observed_mean_mass - max_limit * np.log(10.0)) / (
            np.sqrt(2.0) * observed_mass_sigma
        )

        if x_max > 3.0 or x_min < -3.0:
            #  pylint: disable-next=no-member
            return -(special.erfc(x_min) - special.erfc(x_max)) / 2.0
        #  pylint: disable-next=no-member
        return (special.erf(x_min) - special.erf(x_max)) / 2.0

    def distribution(self, args: List[float], index_lkp: Dict[str, int]):
        return 0

    # TODO UNDERSTAND THIS
    # def spread_point(self, logM: float, z: float, *_) -> float:
    #     """Return the probability of the point argument."""

    #     lnM_obs = self.logM_obs * np.log(10.0)

    #     lnM_mu, sigma = self.richness.cluster_mass_lnM_obs_mu_sigma(logM, z)
    #     x = lnM_obs - lnM_mu
    #     chisq = np.dot(x, x) / (2.0 * sigma**2)
    #     likelihood = np.exp(-chisq) / (np.sqrt(2.0 * np.pi * sigma**2))
    #     return likelihood * np.log(10.0)


class TrueMass(Kernel):
    def __init__(
        self, is_dirac_delta=False, integral_bounds: List[Tuple[float, float]] = None
    ):
        super().__init__(KernelType.mass_proxy, is_dirac_delta, integral_bounds)

    def distribution(self, args: List[float], index_lkp: Dict[str, int]):
        return 1.0
