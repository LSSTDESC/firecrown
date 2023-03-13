"""Cluster Mass Module
abstract class to compute cluster mass function.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, final
from abc import abstractmethod

import pyccl as ccl
import numpy as np



class ClusterMass():
    """Cluster Mass module."""
    def __init__(
        self,
        ccl_hmd,
        ccl_hmf,
        hmf_args: Optional[List[str]] = None
    ):
       
        self.ccl_hmd = ccl_hmd
        self.ccl_hmf = ccl_hmf
        self.hmf_args = hmf_args
        self.logMl = 10.0
        self.logMu = 17.0	
        self.use_proxy = False 
    def compute_mass_function(self, ccl_cosmo: ccl.Cosmology, logM, z) -> float:
        """
        parameters

        ccl_cosmo : pyccl Cosmology
        logM: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.
        reuturn
        -------

        nm : float
            Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        a = 1.0 / (1.0 + z)  # pylint: disable=invalid-name
        mass = 10 ** (logM)
        if self.hmf_args != None:
            hmf = self.ccl_hmf(ccl_cosmo,self.ccl_hmd,self.hmf_args)
        else:
            hmf = self.ccl_hmf(ccl_cosmo,self.ccl_hmd)
        nm = hmf.get_mass_function(ccl_cosmo, mass, a)  # pylint: disable=invalid-name
        return nm
    
    def set_logM_limits(self, logMl, logMu):
        self.logMl = logMl
        self.logMu = logMu
        return None

    @abstractmethod
    def cluster_logM_p(self,logM, z, logM_obs):
        """Computes the logM proxytractmethod"""

    @abstractmethod
    def cluster_logM_intp(self,logM, z):
        """Computes the logM proxy"""
    @abstractmethod
    def cluster_logM_intp_bin(self,logM, z, logM_obs_lower, logM_obs_upper):
        """Computes logMintp"""
