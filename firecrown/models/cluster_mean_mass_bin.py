"""Cluster Mean Mass Module
abstract class to compute cluster mean mass inside bins.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, final

import pyccl as ccl
import numpy as np
import scipy.integrate
from .cluster_abundance_binned import ClusterAbundanceBinned


class ClusterMeanMassInfo:
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
        z_obs_upper: Optional[float] = None,
    ):
        self.ccl_cosmo = ccl_cosmo
        self.logM = logM
        self.z = z
        self.logM_obs = logM_obs
        self.z_obs = z_obs
        self.logM_obs_lower = logM_obs_lower
        self.logM_obs_upper = logM_obs_upper
        self.z_obs_lower = z_obs_lower
        self.z_obs_upper = z_obs_upper


class ClusterMeanMass:
    """Cluster mean_mass module."""

    def __init__(
        self, cluster_mass, cluster_redshift, sky_area: Optional[float] = None, selection_error: Optional[list[bool]] = [False, False]
    ):
        self.cluster_m = cluster_mass
        self.cluster_z = cluster_redshift
        self.sky_area = sky_area
        self.selection_error = selection_error
        self._compute_bin_logM = None
        self._compute_intp_bin_logM_d2logM = None
        self.cluster_abundance_bin = ClusterAbundanceBinned(
            cluster_mass, cluster_redshift, sky_area, selection_error
        )
        self.bin_funcs = self._cluster_mean_mass_bin_funcs()
        self.info = None
        
    def  _cluster_mass_compute_completeness(self, logM, z):
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc*(1.0+z)
        nc = a_nc + b_nc*(1.0+z)
        C =  (logM/log_mc)**nc / ((logM/log_mc)**nc + 1.0)
        return C
    
    def _cluster_mean_mass_z_intp_logM_intp_bin_d2logM(
        self,
        ccl_cosmo,
        logM: float,
        z: float,
        logM_obs_lower,
        logM_obs_upper,
        z_obs_lower,
        z_obs_upper,
    ):
        intp_z = self.cluster_z.cluster_redshift_intp_bin(
            logM, z, z_obs_lower, z_obs_upper
        )
        intp_logM = self.cluster_m.cluster_logM_intp_bin(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * intp_logM * d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_z_intp_logM_intp_bin_d2logM_integrand(
        self, logM: float, z: float
    ):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper
        z_obs_lower = self.info.logM_obs_lower
        z_obs_upper = self.info.logM_obs_upper

        return self._cluster_mean_mass_z_intp_logM_intp_bin_d2logM(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
        )

    def _cluster_mean_mass_z_intp_logM_intp_bin_logM(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterMeanMassInfo(
            ccl_cosmo,
            logM_obs_lower=logM_obs_lower,
            logM_obs_upper=logM_obs_upper,
            z_obs_lower=z_obs_lower,
            z_obs_upper=z_obs_upper,
        )
        integrand = self._cluster_mean_mass_z_intp_logM_intp_bin_d2logM_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        logM = scipy.integrate.dblquad(
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

        N = self.cluster_abundance_bin.compute_bin_N(
            ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
        )
        return logM * DeltaOmega / N

    def _cluster_mean_mass_z_intp_bin_d2logM(
        self, ccl_cosmo, logM: float, z: float, z_obs_lower, z_obs_upper
    ):
        intp_z = self.cluster_z.cluster_redshift_intp_bin(
            logM, z, z_obs_lower, z_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_z_intp_bin_d2logM_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        z_obs_lower = self.info.logM_obs_lower
        z_obs_upper = self.info.logM_obs_upper

        return self._cluster_mean_mass_z_intp_bin_d2logM(
            ccl_cosmo, logM, z, z_obs_lower, z_obs_upper
        )

    def _cluster_mean_mass_z_intp_bin_logM(
        self, ccl_cosmo, logM_lower, logM_upper, z_obs_lower, z_obs_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterMeanMassInfo(
            ccl_cosmo, z_obs_lower=z_obs_lower, z_obs_upper=z_obs_upper
        )

        integrand = self._cluster_mean_mass_z_intp_bin_d2logM_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        logM = scipy.integrate.dblquad(
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

        N = self.cluster_abundance_bin.compute_bin_N(
            ccl_cosmo, logM_lower, logM_upper, z_obs_lower, z_obs_upper
        )

        return logM * DeltaOmega / N

    def _cluster_mean_mass_logM_intp_bin_d2logM(
        self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper
    ):
        intp_logM = self.cluster_m.cluster_logM_intp_bin(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_logM * d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_logM_intp_bin_d2logM_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper

        return self._cluster_mean_mass_logM_intp_bin_d2logM(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper
        )

    def _cluster_mean_mass_logM_intp_bin_logM(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterMeanMassInfo(
            ccl_cosmo, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper
        )

        integrand = self._cluster_mean_mass_logM_intp_bin_d2logM_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        logM = scipy.integrate.dblquad(
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
        N = self.cluster_abundance_bin.compute_bin_N(
            ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
        )

        return logM * DeltaOmega / N
    
    def _cluster_mean_mass_logM_intp_bin_c_d2logM(
        self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper
    ):
        intp_logM = self.cluster_m.cluster_logM_intp_bin(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        complete = self._cluster_mass_compute_completeness(logM,z)
        print(complete)
        return complete * intp_logM * d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_logM_intp_bin_c_d2logM_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper

        return self._cluster_mean_mass_logM_intp_bin_c_d2logM(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper
        )

    def _cluster_mean_mass_logM_intp_bin_c_logM(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterMeanMassInfo(
            ccl_cosmo, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper
        )

        integrand = self._cluster_mean_mass_logM_intp_bin_c_d2logM_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        logM = scipy.integrate.dblquad(
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
        N = self.cluster_abundance_bin.compute_bin_N(
            ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
        )
        print(N, logM)

        return logM * DeltaOmega / N
    def _cluster_mean_mass_d2logM_integrand(self, logM: float, z: float):
        ccl_cosmo = self.info.ccl_cosmo
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        return d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_bin_logM(
        self, ccl_cosmo, logM_lower, logM_upper, z_lower, z_upper
    ):
        # calls the above with logM_obs and z
        self.info = ClusterMeanMassInfo(ccl_cosmo)
        integrand = self._cluster_mean_mass_d2logM_integrand
        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        logM = scipy.integrate.dblquad(
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
        N = self.cluster_abundance_bin.compute_bin_N(
            ccl_cosmo, logM_lower, logM_upper, z_lower, z_upper
        )

        return logM * DeltaOmega / N

    def _cluster_mean_mass_bin_funcs(self):
        if self.cluster_m.use_proxy == False:
            if self.cluster_z.use_proxy == False:
                self._compute_bin_logM = self._cluster_mean_mass_bin_logM
            else:
                self._compute_bin_logM = self._cluster_mean_mass_z_intp_bin_logM
                self._compute_intp_bin_d2n = self._cluster_mean_mass_z_intp_bin_d2logM
        else:
            if self.cluster_z.use_proxy == False:
                if self.selection_error[0]==False:
                    self._compute_bin_logM = self._cluster_mean_mass_logM_intp_bin_logM
                    self._compute_intp_bin_d2logM = (
                        self._cluster_mean_mass_logM_intp_bin_d2logM
                    )
                else:
                    self._compute_bin_logM = self._cluster_mean_mass_logM_intp_bin_c_logM
                    self._compute_intp_bin_d2logM = (
                        self._cluster_mean_mass_logM_intp_bin_c_d2logM
                    )
            else:
                self._compute_bin_logM = (
                    self._cluster_mean_mass_z_intp_logM_intp_bin_logM
                )
                self._compute_intp_bin_d2logM = (
                    self._cluster_mean_mass_z_intp_logM_intp_bin_d2logM
                )
        return True

    def compute_bin_logM(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
    ):
        return self._compute_bin_logM(
            ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
        )

    def compute_intp_bin_d2logM(self, ccl_cosmo, logM, z, **kargs):
        return self._compute_intp_bin_d2logM(ccl_cosmo, logM, z, **kargs)
