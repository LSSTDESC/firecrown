"""Cluster Abundance Binned Module
abstract class to compute cluster abundance in bins of Mass/MassProxy and Redshift/RedshiftProxy.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, final

import pyccl as ccl
import numpy as np
import scipy.integrate
from .cluster_abundance import ClusterAbundance, ClusterAbundanceInfo

class ClusterAbundanceBinned(ClusterAbundance):
    """Cluster Abundance Binned module."""
    def __init__(
    	self,
    	cluster_mass,
		cluster_redshift,
		sky_area: Optional[float] = None
    ):
        super(ClusterAbundanceBinned, self).__init__(cluster_mass, cluster_redshift, sky_area)
        self._compute_bin_N = None
        self._compute_intp_bin_d2n = None

        self.bin_funcs = self._cluster_abundance_bin_funcs()

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
        z_obs_lower = self.info.logM_obs_lower
        z_obs_upper = self.info.logM_obs_upper

        return self._cluster_abundance_z_intp_bin_d2n(ccl_cosmo, logM, z, z_obs_lower, z_obs_upper)


    def _cluster_abundance_z_intp_bin_N(self, ccl_cosmo, logM_lower, logM_upper, z_obs_lower, z_obs_upper):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(ccl_cosmo, z_obs_lower=z_obs_lower, z_obs_upper=z_obs_upper)

        integrand = self._cluster_abundance_z_intp_bin_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: logM_lower,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: logM_upper,
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

        return self._cluster_abundance_logM_intp_bin_d2n(ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper)

    def _cluster_abundance_logM_intp_bin_N(self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(ccl_cosmo, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper)

        integrand = self._cluster_abundance_logM_intp_bin_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        z_lower,
                        z_upper,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega

    def _cluster_abundance_d2n_integrand(self, logM: float, z: float):
        return super(ClusterAbundanceBinned, self)._cluster_abundance_d2n_integrand(logM, z)

    def _cluster_abundance_bin_N(self, ccl_cosmo, logM_lower, logM_upper, z_lower, z_upper):
    #calls the above with logM_obs and z
        self.info = ClusterAbundanceInfo(ccl_cosmo)
        integrand = self._cluster_abundance_d2n_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        z_lower,
                        z_upper,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: logM_lower,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: logM_upper,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega


    def _cluster_abundance_bin_funcs(self):
        if self.cluster_m.use_proxy == False:
            if self.cluster_z.use_proxy == False:
                self._compute_bin_N = self._cluster_abundance_bin_N
            else:
                self._compute_bin_N = self._cluster_abundance_z_intp_bin_N
                self._compute_intp_bin_d2n = self._cluster_abundance_z_intp_bin_d2n
        else:
            if self.cluster_z.use_proxy == False:
                self._compute_bin_N = self._cluster_abundance_logM_intp_bin_N
                self._compute_intp_bin_d2n = self._cluster_abundance_logM_intp_bin_d2n
            else:
                self._compute_bin_N = self._cluster_abundance_z_intp_logM_intp_bin_N
                self._compute_intp_bin_d2n = self._cluster_abundance_z_intp_logM_intp_bin_d2n
        return True


    def compute_bin_N(self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper):
        return self._compute_bin_N(ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper)

    def compute_intp_bin_d2n(self, ccl_cosmo, logM, z, **kargs):
        return self._compute_intp_bin_d2n(ccl_cosmo, logM, z,**kargs)
