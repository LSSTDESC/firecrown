"""Cluster Mass Richness proxy
abstract class to compute cluster mass function.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, final
from abc import abstractmethod

import pyccl as ccl
import numpy as np
from scipy import special
from .cluster_mass import ClusterMass


class ClusterMassRich(ClusterMass):
    """Cluster Mass Richness proxy."""

    def __init__(
        self,
        ccl_hmd,
        ccl_hmf,
        proxy_params,
        pivot_mass,
        pivot_redshift,
        hmf_args: Optional[array] = None,
    ):
        super(ClusterMassRich, self).__init__(ccl_hmd, ccl_hmf, hmf_args)
        self.proxy_params = proxy_params
        self.use_proxy = True
        self.pivot_mass = pivot_mass
        self.pivot_redshift = pivot_redshift
        self.log_pivot_mass = np.log(10**pivot_mass)
        self.log_1_p_pivot_redshift = np.log(1.0 + self.pivot_redshift)

        self.logM_obs_min = 0.0
        self.logM_obs_max = np.inf

    def _cluster_mass_lnM_obs_mu_sigma(self, logM, z):
        mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2 = self.proxy_params
        lnM = np.log(10**logM)

        lnM_obs_mu = (
            mu_p0
            + mu_p1 * (lnM - self.log_pivot_mass)
            + mu_p2 * (np.log(1.0 + z) - self.log_1_p_pivot_redshift)
        )
        sigma = (
            sigma_p0
            + sigma_p1 * (lnM - self.log_pivot_mass)
            + sigma_p2 * (np.log(1.0 + z) - self.log_1_p_pivot_redshift)
        )
        # sigma = abs(sigma)
        return [lnM_obs_mu, sigma]

    def cluster_logM_p(self, logM, z, logM_obs):
        lnM_obs = np.log(10**logM_obs)
        lnM_mu, sigma = self._cluster_mass_lnM_obs_mu_sigma(logM, z)
        x = lnM_obs - lnM_mu
        chisq = np.dot(x, x) / (2 * sigma**2)
        lk = np.exp(-chisq) / (np.sqrt(2.0 * np.pi * sigma**2))
        return lk * np.log(10)


    def cluster_logM_intp(self, logM, z, logM_obs_lower, logM_obs_upper):
        lnM_obs_mu, sigma = self._cluster_mass_lnM_obs_mu_sigma(logM, z)
        x_min = (lnM_obs_mu - np.log(10**logM_obs_lower)) / (np.sqrt(2.0) * sigma)
        x_max = (lnM_obs_mu - np.log(10**logM_obs_upper)) / (np.sqrt(2.0) * sigma)

        if x_max > 4.0:
            return (special.erfc(x_min) - special.erfc(x_max)) / 2.0
        else:
            return (special.erf(x_min) - special.erf(x_max)) / 2.0
