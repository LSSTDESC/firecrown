"""Cluster Redshift Module
abstract class to compute cluster redshift functions.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, final
from abc import abstractmethod

import pyccl as ccl
import numpy as np



class ClusterRedshift():
    """Cluster Redshift module."""
    def __init__(
        self,
    ):
        self.zl = 0.0
        self.zu = np.inf


    def compute_differential_comoving_volume(self, ccl_cosmo: ccl.Cosmology, z) -> float:
        """
        parameters
        ccl_cosmo : pyccl Cosmology
        z : float
            Cluster Redshift.
        reuturn
        -------
        dv : float
            Differential Comoving Volume at z in units of Mpc^3 (comoving).
        """
        a = 1.0 / (1.0 + z)  # pylint: disable=invalid-name
        # pylint: disable-next=invalid-name
        da = ccl.background.angular_diameter_distance(ccl_cosmo, a)
        E = ccl.background.h_over_h0(ccl_cosmo, a)  # pylint: disable=invalid-name
        dV = (  # pylint: disable=invalid-name
            ((1.0 + z) ** 2)
            * (da**2)
            * ccl.physical_constants.CLIGHT_HMPC
            / ccl_cosmo["h"]
            / E
        )
        return dV

    def set_redshift_limits(self, zl, zu):
        self.zl = zl
        self.zu = zu
        return None

    @abstractmethod
    def cluster_z_p(self, ccl_cosmo,logM, z, z_obs, z_obs_params):
        """Computes the logM proxy"""

    @abstractmethod
    def cluster_z_intp(self, ccl_cosmo,logM, z, z_obs, z_obs_params):
        """Computes the logM proxy"""










