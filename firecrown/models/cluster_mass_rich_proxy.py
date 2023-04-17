"""Cluster Mass Richness proxy
abstract class to compute cluster mass function.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import Optional, List, final

import numpy as np
from scipy import special

from ..parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
)
from .cluster_mass import ClusterMass
from .. import parameters


class ClusterMassRich(ClusterMass):
    """Cluster Mass Richness proxy."""

    def __init__(
        self,
        ccl_hmd,
        ccl_hmf,
        pivot_mass,
        pivot_redshift,
        hmf_args: Optional[List[str]] = None,
    ):
        super().__init__(ccl_hmd, ccl_hmf, hmf_args)
        self.use_proxy = True
        self.pivot_mass = pivot_mass
        self.pivot_redshift = pivot_redshift
        self.log_pivot_mass = np.log(10**pivot_mass)
        self.log_1_p_pivot_redshift = np.log(1.0 + self.pivot_redshift)

        # Updatable parameters
        self.mu_p0 = parameters.create()
        self.mu_p1 = parameters.create()
        self.mu_p2 = parameters.create()
        self.sigma_p0 = parameters.create()
        self.sigma_p1 = parameters.create()
        self.sigma_p2 = parameters.create()

        self.logM_obs_min = 0.0  # pylint: disable-msg=invalid-name
        self.logM_obs_max = np.inf  # pylint: disable-msg=invalid-name

    @final
    def _update_cluster_mass(self, params: ParamsMap):
        """Perform any updates necessary after the parameters have being updated.

        This implementation has nothing to do."""

    @final
    def _reset_cluster_mass(self) -> None:
        """Reset the ClusterMass object.

        This implementation has nothing to do."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    #  pylint: disable-next=invalid-name
    def _cluster_mass_lnM_obs_mu_sigma(self, logM, z):
        lnM = np.log(10**logM)  # pylint: disable-msg=invalid-name

        lnM_obs_mu = (  # pylint: disable-msg=invalid-name
            self.mu_p0
            + self.mu_p1 * (lnM - self.log_pivot_mass)
            + self.mu_p2 * (np.log(1.0 + z) - self.log_1_p_pivot_redshift)
        )
        sigma = (
            self.sigma_p0
            + self.sigma_p1 * (lnM - self.log_pivot_mass)
            + self.sigma_p2 * (np.log(1.0 + z) - self.log_1_p_pivot_redshift)
        )
        # sigma = abs(sigma)
        return [lnM_obs_mu, sigma]

    def cluster_logM_p(self, logM, z, logM_obs):  # pylint: disable-msg=invalid-name
        lnM_obs = np.log(10**logM_obs)  # pylint: disable-msg=invalid-name
        # pylint: disable-next=invalid-name
        lnM_mu, sigma = self._cluster_mass_lnM_obs_mu_sigma(logM, z)
        x = lnM_obs - lnM_mu  # pylint: disable-msg=invalid-name
        chisq = np.dot(x, x) / (2 * sigma**2)
        likelihood = np.exp(-chisq) / (np.sqrt(2.0 * np.pi * sigma**2))
        return likelihood * np.log(10)

    def cluster_logM_intp(self, logM, z, logM_obs_lower, logM_obs_upper):
        #  pylint: disable-next=invalid-name
        lnM_obs_mu, sigma = self._cluster_mass_lnM_obs_mu_sigma(logM, z)
        x_min = (lnM_obs_mu - np.log(10**logM_obs_lower)) / (np.sqrt(2.0) * sigma)
        x_max = (lnM_obs_mu - np.log(10**logM_obs_upper)) / (np.sqrt(2.0) * sigma)

        if x_max > 4.0:
            return (special.erfc(x_min) - special.erfc(x_max)) / 2.0
        else:
            return (special.erf(x_min) - special.erf(x_max)) / 2.0
