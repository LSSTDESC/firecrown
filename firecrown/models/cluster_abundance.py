"""Cluster Abundance Module 
abstract class to compute cluster mass function and its abundance.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, final

import pyccl as ccl
import numpy as np


class ClusterAbundanceInfo():
	def __init__(
		ccl_cosmo, 
		logM: Optional[float] = None,
		z: Optional[float] = None,
		logM_obs_params: Optional[List[float]] = None,
		z_obs_params: Optional[List[float]] = None,
	):

		self.ccl_cosmo = ccl_cosmo
		self.logM = logM
		self.z = z
		self.logM_obs_params = logM_obs_params
		self.z_obs_params = z_obs_params
class ClusterAbundance():
    """Cluster Abundance module."""
    def __init__(
    	self,
    	halo_mass_function,
		logM_p: Optional[ClusterMassProxy] = None,
		redshift_proxy: Optional [ClusterRedshiftProxy] = None,
    ):
		self.hmf = halo_mass_function
		self.logM_p = logM_proxy
		self.z_p = redshift_proxy
		self.define_funcs = self._cluster_abundance_funcs(self.mass_p, self.z_p)
		self.info = None

	def _cluster_abundance_z_p_logM_p_d2n_integrand(self, logM: float, z: float):
	"""Define integrand for the case when we have proxy for redshift and mass.
	The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
	"""
		ccl_cosmo = self.info.ccl_cosmo
		p_z_zt = self.z_p.cluster_redshift_p(ccl_cosmo, logM, z)

	def _cluster_abundance_z_p_logM_p_d2n(self, ccl_cosmo, logM_obs: float, logM_obs_params, z_obs: float, z_obs_params):
	"""Integral of integrand when we have the two proxies redshift and mass"""
	#calls the above with logM_obs and z_obs
		self.info = ClusterAbundanceInfo(ccl_cosmo, logM_obs_params=logM_obs_params, z_obs_params=z_obs_params)

	def _cluster_abundance_z_p_d2n_integrand(self, logM: float, z: float):
		ccl_cosmo = self.info.ccl_cosmo
		logM = self.info.logM
		logM_obs_params = self.info.logM_obs_params
		z_obs_params = self.info.z_obs_params

	def _cluster_abundance_z_p_d2n(self, ccl_cosmo, logM: float, z_obs: float):
	#calls the above with logM and z_obs
		self.info = ClusterAbundanceInfo(ccl_cosmo, logM=logM)

    def _cluster_abundance_logM_p_d2n_integrand(self, logM: float):
		ccl_cosmo = self.info.ccl_cosmo
        z = self.info.z
		
    def _cluster_abundance_logM_p_d2n(self, ccl_cosmo, logM_obs: float, z: float):
    #calls the above with logM_obs and z
		self.info = ClusterAbundanceInfo(ccl_cosmo,z=z)
		
	
	def _cluster_abundance_d2n(self, logM: float, z: float):
		

	def 





