"""Cluster Mass Richness proxy module

Define the Cluster Mass Richness proxy module and its arguments.
"""
from typing import Tuple

import numpy as np
from scipy import special
from ..parameters import ParamsMap
from .. import parameters
from .kernel import Kernel


class MassObservable(Kernel):
    def __init__(self, params: ParamsMap):
        """Initialize the ClusterMassRich object."""
        self.pivot_mass = params["pivot_mass"]
        self.pivot_redshift = params["pivot_redshift"]
        self.pivot_mass = self.pivot_mass * np.log(10.0)

        self.min_mass = params["min_mass"]  # 13
        self.max_mass = params["max_mass"]  # 16

        super().__init__()


class TrueMass(MassObservable):
    def __init__(self, params: ParamsMap):
        super().__init__(params)


class MassRichnessMuSigma(MassObservable):
    def __init__(self, params: ParamsMap, bounds):
        super().__init__(params)

        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        # Updatable parameters
        self.mu_p0 = parameters.create()
        self.mu_p1 = parameters.create()
        self.mu_p2 = parameters.create()
        self.sigma_p0 = parameters.create()
        self.sigma_p1 = parameters.create()
        self.sigma_p2 = parameters.create()

    @staticmethod
    def observed_mass(
        pivot_mass, log1p_pivot_redshift, p: Tuple[float, float, float], mass, z
    ):
        """Return observed quantity corrected by redshift and mass."""

        ln_mass = mass * np.log(10)
        delta_ln_mass = ln_mass - pivot_mass
        delta_z = np.log1p(z) - log1p_pivot_redshift

        return p[0] + p[1] * delta_ln_mass + p[2] * delta_z

    def probability(self, mass, z):
        observed_mean_mass = MassRichnessMuSigma.observed_mass(
            self.pivot_mass,
            self.log1p_pivot_redshift,
            (self.mu_p0, self.mu_p1, self.mu_p2),
            mass,
            z,
        )
        observed_mass_sigma = MassRichnessMuSigma.observed_mass(
            self.pivot_mass,
            self.log1p_pivot_redshift,
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            mass,
            z,
        )

        x_min = (observed_mean_mass - self.min_obs_mass * np.log(10.0)) / (
            np.sqrt(2.0) * observed_mass_sigma
        )
        x_max = (observed_mean_mass - self.max_obs_mass * np.log(10.0)) / (
            np.sqrt(2.0) * observed_mass_sigma
        )

        if x_max > 3.0 or x_min < -3.0:
            #  pylint: disable-next=no-member
            return -(special.erfc(x_min) - special.erfc(x_max)) / 2.0
        #  pylint: disable-next=no-member
        return (special.erf(x_min) - special.erf(x_max)) / 2.0

    # TODO UNDERSTAND THIS
    # def spread_point(self, logM: float, z: float, *_) -> float:
    #     """Return the probability of the point argument."""

    #     lnM_obs = self.logM_obs * np.log(10.0)

    #     lnM_mu, sigma = self.richness.cluster_mass_lnM_obs_mu_sigma(logM, z)
    #     x = lnM_obs - lnM_mu
    #     chisq = np.dot(x, x) / (2.0 * sigma**2)
    #     likelihood = np.exp(-chisq) / (np.sqrt(2.0 * np.pi * sigma**2))
    #     return likelihood * np.log(10.0)
