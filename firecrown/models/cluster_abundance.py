"""Cluster Abundance Module 
abstract class to compute cluster abundance.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, final

import pyccl as ccl
import numpy as np
import scipy.integrate

class ClusterAbundanceInfo():
    def __init__(
        self,
        ccl_cosmo,
        logM: Optional[float] = None,
        z: Optional[float] = None,
        logM_obs: Optional[float] = None,
        z_obs: Optional[float] = None,
        logM_obs_lower: Optional[float] = None,
        logM_obs_upper: Optional[float] = None,
        z_obs_lower: Optional[float] = None,
        z_obs_upper: Optional[float] = None
    ):

        self.ccl_cosmo = ccl_cosmo
        self.logM = logM
        self.z = z
        self.logM_obs = logM_obs
        self.z_obs = z_obs
        self.logM_obs_lower = logM_obs_lower,
        self.logM_obs_upper = logM_obs_upper,
        self.z_obs_lower = z_obs_lower,
        self.z_obs_upper = z_obs_upper	

class ClusterAbundance():
    """Cluster Abundance module."""
    def __init__(
    	self,
    	cluster_mass,
		cluster_redshift,
		sky_area: Optional[float] = None
    ):
        self.cluster_m = cluster_mass
        self.cluster_z = cluster_redshift
        self.sky_area = sky_area
        self._compute_N = None
        self._compute_intp_d2n = None
        self._compute_bin_N = None
        self._compute_intp_bin_d2n = None

        self.funcs = self._cluster_abundance_funcs()
        self.info = None



    def _cluster_abundance_z_p_logM_p_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_params = self.cluster_m.logM_obs_params
        z_obs_params = self.cluster_z.z_obs_params
        z_obs = self.info.z_obs
        logM_obs = self.info.logM_obs

        p_z = self.cluster_z.cluster_redshift_p(logM, z, z_obs)
        p_logM = self.cluster_m.cluster_logM_p(logM, z, logM_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z) 		

        return p_z * p_logM * d2NdzdlogM * dvdz
		 
    def cluster_abundance_z_p_logM_p_d2n(self, ccl_cosmo, logM_obs: float, z_obs: float):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(ccl_cosmo, logM_obs=logM_obs, z_obs=z_obs)

        integrand = self._cluster_abundance_z_p_logM_p_d2n_integrand
        nm = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return nm
    

    def _cluster_abundance_z_p_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z)dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM = self.info.logM
        z_obs = self.info.z_obs
		

        p_z = self.cluster_z.cluster_redshift_p(logM, z, z_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return p_z * d2NdzdlogM * dvdz


    def cluster_abundance_z_p_d2n(self, ccl_cosmo, logM: float, z_obs: float):
	#calls the above with logM and z_obs
        self.info = ClusterAbundanceInfo(ccl_cosmo, logM=logM, z_obs=z_obs)

        integrand = self._cluster_abundance_z_p_d2n_integrand
        nm = scipy.integrate.quad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                    )[0]
        return nm
    def _cluster_abundance_logM_p_d2n_integrand(self, logM: float):
        ccl_cosmo = self.info.ccl_cosmo
        z = self.info.z
        logM_obs = self.info.logM_obs

        p_logM = self.cluster_m.cluster_logM_p(logM, z, logM_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return p_logM * d2NdzdlogM * dvdz

    def cluster_abundance_logM_p_d2n(self, ccl_cosmo, logM_obs: float, z: float):
    #calls the above with logM_obs and z
        self.info = ClusterAbundanceInfo(ccl_cosmo, z=z, logM_obs=logM_obs)

        integrand = self._cluster_abundance_logM_p_d2n_integrand
        nm = scipy.integrate.quad(
                        integrand,
                        self.cluster_m.logMl,
                        self.cluster_m.logMu,
                    )[0]
        return nm	
	
    def _cluster_abundance_d2n_integrand(self, logM: float, z: float):
        ccl_cosmo = self.info.ccl_cosmo
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        return d2NdzdlogM * dvdz

    def _cluster_abundance_d2n(self, ccl_cosmo, logM: float, z: float):
        self.info = ClusterAbundanceInfo(ccl_cosmo)
        return self._cluster_abundance_d2n_integrand( logM, z)

    def _cluster_abundance_z_intp_logM_intp_d2n(self, ccl_cosmo, logM: float, z: float):
        
        intp_z = self.cluster_z.cluster_redshift_intp(logM, z)
        intp_logM = self.cluster_m.cluster_logM_intp(logM, z)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * intp_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_z_intp_logM_intp_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo

        return self._cluster_abundance_z_intp_logM_intp_d2n(ccl_cosmo, logM, z)
	
	 
    def _cluster_abundance_z_intp_logM_intp_N(self, ccl_cosmo):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(ccl_cosmo)

        integrand = self._cluster_abundance_z_intp_logM_intp_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2        
        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega
    
    def _cluster_abundance_z_intp_logM_intp_bin_d2n(self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper):

        intp_z = self.cluster_z.cluster_redshift_intp_bin(logM, z, z_obs_lower, z_obs_upper)
        intp_logM = self.cluster_m.cluster_logM_intp_bin(logM, z, logM_obs_lower, logM_obs_upper)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * intp_logM * d2NdzdlogM * dvdz
    def _cluster_abundance_z_intp_logM_intp_bin_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper
        z_obs_lower = self.info.logM_obs_lower
        z_obs_upper = self.info.logM_obs_upper

        return self._cluster_abundance_z_intp_logM_intp_bin_d2n(ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper)


    def _cluster_abundance_z_intp_logM_intp_bin_N(self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(ccl_cosmo, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper, z_obs_lower=z_obs_lower, z_obs_upper=z_obs_upper)
        integrand = self._cluster_abundance_z_intp_logM_intp_bin_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega


    def _cluster_abundance_z_intp_d2n(self, ccl_cosmo, logM: float, z: float):


        intp_z = self.cluster_z.cluster_redshift_intp(logM, z)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * d2NdzdlogM * dvdz

    def _cluster_abundance_z_intp_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z)dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
		
        return self._cluster_abundance_z_intp_d2n(ccl_cosmo, logM, z)


    def _cluster_abundance_z_intp_N(self, ccl_cosmo):
	#calls the above with logM and z_obs

       self.info = ClusterAbundanceInfo(ccl_cosmo)

       integrand = self._cluster_abundance_z_intp_d2n_integrand
       DeltaOmega = self.sky_area * np.pi**2 / 180**2

       N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]

       return N * DeltaOmega

    def _cluster_abundance_z_intp_bin_d2n(self, ccl_cosmo, logM: float, z: float, z_obs_lower, z_obs_upper):

        intp_z = self.cluster_z.cluster_redshift_intp_bin(logM, z, z_obs_lower, z_obs_upper)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * d2NdzdlogM * dvdz

    def _cluster_abundance_z_intp_bin_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper
        z_obs_lower = self.info.logM_obs_lower
        z_obs_upper = self.info.logM_obs_upper

        return self._cluster_abundance_z_intp_bin_d2n(ccl_cosmo, logM, z, z_obs_lower, z_obs_upper)


    def _cluster_abundance_z_intp_bin_N(self, ccl_cosmo, z_obs_lower, z_obs_upper):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(ccl_cosmo, z_obs_lower=z_obs_lower, z_obs_upper=z_obs_upper)

        integrand = self._cluster_abundance_z_intp_bin_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega


    def _cluster_abundance_logM_intp_d2n(self, ccl_cosmo, logM: float, z: float):
        intp_logM = self.cluster_m.cluster_logM_intp(logM, z)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_logM * d2NdzdlogM * dvdz
    
    def _cluster_abundance_logM_intp_d2n_integrand(self, logM: float, z):

        ccl_cosmo = self.info.ccl_cosmo

        return self._cluster_abundance_logM_intp_d2n(ccl_cosmo, logM, z)

    def _cluster_abundance_logM_intp_N(self, ccl_cosmo):
    #calls the above with logM_obs and z
        self.info = ClusterAbundanceInfo(ccl_cosmo)
        integrand = self._cluster_abundance_logM_intp_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega

    def _cluster_abundance_logM_intp_bin_d2n(self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper):

        intp_logM = self.cluster_m.cluster_logM_intp_bin(logM, z, logM_obs_lower, logM_obs_upper)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_logM * d2NdzdlogM * dvdz
    def _cluster_abundance_logM_intp_bin_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper
        z_obs_lower = self.info.logM_obs_lower
        z_obs_upper = self.info.logM_obs_upper

        return self._cluster_abundance_z_intp_logM_intp_bin_d2n(ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper)

    def _cluster_abundance_logM_intp_bin_N(self, ccl_cosmo, logM_obs_lower, logM_obs_upper):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(ccl_cosmo, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper)
        integrand = self._cluster_abundance_logM_intp_bin_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega

    def _cluster_abundance_N(self, ccl_cosmo):
    #calls the above with logM_obs and z
        self.info = ClusterAbundanceInfo(ccl_cosmo)
        integrand = self._cluster_abundance_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega

	
    def _cluster_abundance_funcs(self):
        if self.cluster_m.use_proxy == False: 
            if self.cluster_z.use_proxy == False:
                self._compute_N = self._cluster_abundance_N
                self._compute_intp_d2n = self._cluster_abundance_d2n
            else:
                self._compute_N = self._cluster_abundance_z_intp_N
                self._compute_intp_d2n = self._cluster_abundance_z_intp_d2n
                self._compute_bin_N = self._cluster_abundance_z_intp_bin_N
                self._compute_intp_bin_d2n = self._cluster_abundance_z_intp_bin_d2n
        else:
            if self.cluster_z.use_proxy == False:
                self._compute_N = self._cluster_abundance_logM_intp_N
                self._compute_intp_d2n = self._cluster_abundance_logM_intp_d2n
                self._compute_bin_N = self._cluster_abundance_logM_intp_bin_N
                self._compute_intp_bin_d2n = self._cluster_abundance_logM_intp_bin_d2n
            else:
                self._compute_N = self._cluster_abundance_z_intp_logM_intp_N
                self._compute_intp_d2n = self._cluster_abundance_z_intp_logM_intp_d2n
                self._compute_bin_N = self._cluster_abundance_z_intp_logM_intp_bin_N
                self._compute_intp_bin_d2n = self._cluster_abundance_z_intp_logM_intp_bin_d2n
        return True
 
    def compute_N(self, ccl_cosmo):    
        return self._compute_N(ccl_cosmo)

    def compute_intp_d2n(self, ccl_cosmo, logM, z):
        return self._compute_intp_d2n(ccl_cosmo, logM, z)

    def compute_bin_N(self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper):
        return self._compute_bin_N(ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper)

    def compute_intp_bin_d2n(self, ccl_cosmo, logM, z, **kargs):
        return self._compute_intp_bin_d2n(ccl_cosmo, logM, z, **kargs)




