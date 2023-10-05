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

        self.min_mass = params["min_mass"]
        self.max_mass = params["max_mass"]

        self.min_obs_mass = 0.0
        self.max_obs_mass = np.inf
        super().__init__()


class TrueMass(MassObservable):
    def __init__(self, params: ParamsMap):
        super().__init__(params)


class MassRichnessMuSigma(MassObservable):
    def __init__(self, params: ParamsMap):
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
    def cluster_mass_parameters_function(
        log_pivot_mass, log1p_pivot_redshift, p: Tuple[float, float, float], logM, z
    ):
        """Return observed quantity corrected by redshift and mass."""

        lnM = logM * np.log(10)
        Delta_lnM = lnM - log_pivot_mass
        Delta_z = np.log1p(z) - log1p_pivot_redshift

        return p[0] + p[1] * Delta_lnM + p[2] * Delta_z

    def cluster_mass(self, mass, redshift):
        return [
            MassRichnessMuSigma.cluster_mass_parameters_function(
                self.pivot_mass,
                self.log1p_pivot_redshift,
                (self.mu_p0, self.mu_p1, self.mu_p2),
                mass,
                redshift,
            ),
            MassRichnessMuSigma.cluster_mass_parameters_function(
                self.pivot_mass,
                self.log1p_pivot_redshift,
                (self.sigma_p0, self.sigma_p1, self.sigma_p2),
                mass,
                redshift,
            ),
        ]

    def distribution(self, mass, z, mass_proxy, z_proxy):
        lnM_obs_mu, sigma = self.cluster_mass(mass, z)

        x_min = (lnM_obs_mu - self.min_obs_mass * np.log(10.0)) / (np.sqrt(2.0) * sigma)
        x_max = (lnM_obs_mu - self.max_obs_mass * np.log(10.0)) / (np.sqrt(2.0) * sigma)

        if x_max > 3.0 or x_min < -3.0:
            #  pylint: disable-next=no-member
            return -(special.erfc(x_min) - special.erfc(x_max)) / 2.0
        #  pylint: disable-next=no-member
        return (special.erf(x_min) - special.erf(x_max)) / 2.0

    # TODO UNDERSTAND THIS
    def spread_point(self, logM: float, z: float, *_) -> float:
        """Return the probability of the point argument."""

        lnM_obs = self.logM_obs * np.log(10.0)

        lnM_mu, sigma = self.richness.cluster_mass_lnM_obs_mu_sigma(logM, z)
        x = lnM_obs - lnM_mu
        chisq = np.dot(x, x) / (2.0 * sigma**2)
        likelihood = np.exp(-chisq) / (np.sqrt(2.0 * np.pi * sigma**2))
        return likelihood * np.log(10.0)
