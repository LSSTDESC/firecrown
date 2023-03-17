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
        self, cluster_mass, cluster_redshift, sky_area: Optional[float] = None, selection_error: Optional[list[bool]] = [False, False]
    ):
        super(ClusterAbundanceBinned, self).__init__(
            cluster_mass, cluster_redshift, sky_area
        )
        self._compute_bin_N = None
        self._compute_intp_bin_d2n = None
        self.selection_error = selection_error
        self.bin_funcs = self._cluster_abundance_bin_funcs()

    def _cluster_abundance_compute_purity(self, logM_obs, z):
        ln_r = np.log(10**logM_obs)
        a_nc = np.log(10)*0.8612
        b_nc = np.log(10)*0.3527
        a_rc = 2.2183
        b_rc = -0.6592
        nc = a_nc + b_nc*(1.0+z)
        ln_rc = a_rc + b_rc*(1.0+z)
        purity = (ln_r/ln_rc)**nc / ((ln_r/ln_rc)**nc + 1.0)
        return purity

    def  _cluster_abundance_compute_completeness(self, logM, z):
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc*(1.0+z)
        nc = a_nc + b_nc*(1.0+z)
        C =  (logM/log_mc)**nc / ((logM/log_mc)**nc + 1.0)
        return C

    def _cluster_abundance_z_intp_logM_intp_bin_d2n(
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

        return intp_z * intp_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_z_intp_logM_intp_bin_d2n_integrand(
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

        return self._cluster_abundance_z_intp_logM_intp_bin_d2n(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
        )

    def _cluster_abundance_z_intp_logM_intp_bin_N(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(
            ccl_cosmo,
            logM_obs_lower=logM_obs_lower,
            logM_obs_upper=logM_obs_upper,
            z_obs_lower=z_obs_lower,
            z_obs_upper=z_obs_upper,
        )
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

    def _cluster_abundance_z_intp_bin_d2n(
        self, ccl_cosmo, logM: float, z: float, z_obs_lower, z_obs_upper
    ):
        intp_z = self.cluster_z.cluster_redshift_intp_bin(
            logM, z, z_obs_lower, z_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * d2NdzdlogM * dvdz

    def _cluster_abundance_z_intp_bin_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        z_obs_lower = self.info.logM_obs_lower
        z_obs_upper = self.info.logM_obs_upper

        return self._cluster_abundance_z_intp_bin_d2n(
            ccl_cosmo, logM, z, z_obs_lower, z_obs_upper
        )

    def _cluster_abundance_z_intp_bin_N(
        self, ccl_cosmo, logM_lower, logM_upper, z_obs_lower, z_obs_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(
            ccl_cosmo, z_obs_lower=z_obs_lower, z_obs_upper=z_obs_upper
        )

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

    def _cluster_abundance_logM_intp_bin_d2n(
        self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper
    ):
        intp_logM = self.cluster_m.cluster_logM_intp_bin(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_logM_intp_bin_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper
        # print(logM_obs_lower, logM_obs_upper)
        return self._cluster_abundance_logM_intp_bin_d2n(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper
        )

    def _cluster_abundance_logM_intp_bin_N(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(
            ccl_cosmo, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper
        )

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

    def _cluster_abundance_logM_intp_bin_c_d2n(
        self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper
    ):
        intp_logM = self.cluster_m.cluster_logM_intp_bin(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        complete = self._cluster_abundance_compute_completeness(logM, z)
        return complete * intp_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_logM_intp_bin_c_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper
        # print(logM_obs_lower, logM_obs_upper)
        return self._cluster_abundance_logM_intp_bin_c_d2n(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper
        )
    def _cluster_abundance_logM_intp_bin_c_N(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(
            ccl_cosmo, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper
        )

        integrand = self._cluster_abundance_logM_intp_bin_c_d2n_integrand
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
    def _cluster_abundance_logM_intp_bin_cp_d2n(
        self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper
    ):
        intp_logM = self.cluster_m.cluster_logM_intp_bin(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        purity = self._cluster_abundance_compute_purity((logM_obs_upper - logM_obs_lower)/2.0, z)
        complete = self._cluster_abundance_compute_completeness(logM, z)
        return complete / purity * intp_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_logM_intp_bin_cp_d2n_integrand(self, logM: float, z: float):
        """Define integrand for the case when we have proxy for redshift and mass.
        The integrand is d2n/dlogMdz * P(z_obs|logM, z) * P(logM_obs|logM, z) dlogM dz.
        """
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper
        # print(logM_obs_lower, logM_obs_upper)
        return self._cluster_abundance_logM_intp_bin_cp_d2n(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper
        )
    def _cluster_abundance_logM_intp_bin_cp_N(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):
        """Integral of integrand when we have the two proxies redshift and mass"""

        self.info = ClusterAbundanceInfo(
            ccl_cosmo, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper
        )

        integrand = self._cluster_abundance_logM_intp_bin_cp_d2n_integrand
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
        return super(ClusterAbundanceBinned, self)._cluster_abundance_d2n_integrand(
            logM, z
        )

    def _cluster_abundance_bin_N(
        self, ccl_cosmo, logM_lower, logM_upper, z_lower, z_upper
    ):
        # calls the above with logM_obs and z
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
                if self.selection_error[0]  == False:
                    if self.selection_error[1] == False:
                        self._compute_bin_N = self._cluster_abundance_logM_intp_bin_N
                        self._compute_intp_bin_d2n = self._cluster_abundance_logM_intp_bin_d2n
                else:
                    if self.selection_error[1] == True:
                        self._compute_bin_N = self._cluster_abundance_logM_intp_bin_cp_N
                        self._compute_intp_bin_d2n = self._cluster_abundance_logM_intp_bin_cp_d2n
                    else:
                        self._compute_bin_N = self._cluster_abundance_logM_intp_bin_c_N
                        self._compute_intp_bin_d2n = self._cluster_abundance_logM_intp_bin_c_d2n
            else:
                self._compute_bin_N = self._cluster_abundance_z_intp_logM_intp_bin_N
                self._compute_intp_bin_d2n = (
                    self._cluster_abundance_z_intp_logM_intp_bin_d2n
                )
        return True

    def compute_bin_N(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
    ):
        return self._compute_bin_N(
            ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
        )

    def compute_intp_bin_d2n(self, ccl_cosmo, logM, z, **kargs):
        return self._compute_intp_bin_d2n(ccl_cosmo, logM, z, **kargs)
