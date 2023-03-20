r"""Cluster Mean Mass Module
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
    r"""Cluster mean_mass module.
        Atributes
        __________
        cluster_mass: ClusterMass object
            Dictates whether to use a mass proxy or not,\
            which mass function and other cluster functions\
            that mostly depend on the cluster mass.
        cluster_redshift: Cluster Redshift object
            Dictates whether to use a redshift proxy or not,\
            how to compute the comoving volume and other cluster functions\
            that mostly depend on the cluster redshift.
        sky_area: float
            Area of the sky from the survey.
        selection_error: array-like, bool
            If set to [True, True], uses a funciton to\
            compute the completeness and purity of the\
            cluster catalog respectively.
    """

    def __init__(
        self,
        cluster_mass,
        cluster_redshift,
        sky_area: Optional[float] = None,
        selection_error: Optional[list[bool]] = [False, False],
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

    def _cluster_mass_compute_completeness(self, logM, z):
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc * (1.0 + z)
        nc = a_nc + b_nc * (1.0 + z)
        C = (logM / log_mc) ** nc / ((logM / log_mc) ** nc + 1.0)
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
        r"""Compute d2m for the case when we have proxy for redshift and mass in bins of $logM_obs$ and $z_obs$.
        The integrand is given by
        .. math::
            d2m(logM, z)^{\alpha \beta} = \int_{logM_obs^\alpha}^{logM_obs^{\alpha+1}}\int_{z_obs^\beta}^{z_obs^{\beta+1}}\frac{d2n}{dlogMdz} logM P(z_obs|logM, z)  P(logM_obs|logM, z) \frac{dv}{dz} dlogM_obs dz_obs.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology.
        logM: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2m^{\alpha \beta}: float
            integrand of the cluster mean mass integral in bins of observed mass and observed redshift.
        """
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
        r"""Computes the integral of $d2n(logM, z)^{\alpha \beta}$ over the true values of mass and redshift\
        for the bins of $logM_obs^{\alpha}$ and $z_obs^{\beta}$ , that is     
        .. math::
            logM^{\alpha \beta} = \frac{1}{N^{\alpha \beta}}\Omega \int_{logM_min}^{logM_max}\int_{z_min}^{z_max}\frac{d2m^{\alpha \beta}}{dlogMdz} \frac{dv}{dz} dlogM dz.

        In the above, we utilize the analitical integral of the proxies.
        
        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology
        logM: float
            Observed cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Observed cluster redshift.

        return
        ______
        logM^{\alpha \beta}: float
            Cluster mean mass in the interval [logM_lower, logM_upper], [z_lower, z_min], [logM_obs^{\alpha}, logM_obs^{\alpha+1}] and [z_obs^{\beta}, z_obs^{\beta}].
        """
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
        r"""Compute d2m for the case when we have proxy for redshift and true mass mass in bins of $z_obs$.
        .. math::
            d2m(logM, z)^{\beta} = \int_{z_obs^\beta}^{z_obs^{\beta+1}}\frac{d2n}{dlogMdz} logM P(z_obs|logM, z) \frac{dv}{dz} dz_obs.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology.
        logM: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2m^\beta: float
            integrand of the cluster mean mass integral in bin of observed redshift.
        """
        intp_z = self.cluster_z.cluster_redshift_intp_bin(
            logM, z, z_obs_lower, z_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_z_intp_bin_d2logM_integrand(self, logM: float, z: float):
        ccl_cosmo = self.info.ccl_cosmo
        z_obs_lower = self.info.logM_obs_lower
        z_obs_upper = self.info.logM_obs_upper

        return self._cluster_mean_mass_z_intp_bin_d2logM(
            ccl_cosmo, logM, z, z_obs_lower, z_obs_upper
        )

    def _cluster_mean_mass_z_intp_bin_logM(
        self, ccl_cosmo, logM_lower, logM_upper, z_obs_lower, z_obs_upper
    ):
        r"""Computes the integral of $d2m(logM, z)^{\beta}$ over the true values of mass and redshift\
        for the bins of $logM^{\alpha}$ and $z_obs^{\beta}$  , that is  
        .. math::
            logM^{\alpha \beta} = \Omega \frac{1}{N^{\alpha \beta}}\int_{logM^{\alpha}}^{logM^{\alpha+1}}\int_{z_obs^\beta}^{z_obs^\beta}\frac{d2m^{\beta}}{dlogMdz} \frac{dv}{dz} dlogM dz.

        In the above, we utilize the analitical integral of the redshift proxy.
        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology

        return
        ______
        logM^{\alpha \beta}: float
            Cluster mean mass in the interval [logM^\alpha, logM^{\alpha+1}], [z_lower, z_min] and [z_obs^{\beta}, z_obs^{\beta +1 }].
        """

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
        r"""Compute d2m for the case when we have proxy for mass and true redshift in bins of $logM_obs$.
        .. math::
            d2m(logM, z)^{\alpha} = \int_{logM_obs^\alpha}^{logM_obs^{\alpha+1}}\frac{d2n}{dlogMdz} logM P(logM_obs|logM, z) \frac{dv}{dz} dlogM_obs.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology.
        logM: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2m^\alpha: float
            Integrand of the cluster mean mass integral in bin of observed mass.
        """
        intp_logM = self.cluster_m.cluster_logM_intp_bin(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_logM * d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_logM_intp_bin_d2logM_integrand(self, logM: float, z: float):
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper

        return self._cluster_mean_mass_logM_intp_bin_d2logM(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper
        )

    def _cluster_mean_mass_logM_intp_bin_logM(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):
        r"""Computes the integral of $d2m(logM, z)^{\alpha}$ over the true values of mass and redshift\
        for the bins of $logM_obs^{\alpha}$ and $z^{\beta}$, that is  
        .. math::
            logM^{\alpha \beta} = \frac{1}{N^{\alpha \beta}}\Omega \int_{logM_obs^{\alpha}}^{logM_obs^{\alpha+1}}\int_{z^\beta}^{z^\beta} \frac{d2m^{\beta}}{dlogMdz} \frac{dv}{dz} dlogM dz.

        In the above, we utilize the analitical integral of the mass proxy.
        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology

        return
        ______
        logM^{\alpha \beta}: float
            Cluster mean mass in the interval [logM_obs^\alpha, logM_obs^{\alpha+1}], [logM_lower, logM_min] and [z^{\beta}, z^{\beta +1 }].
        """
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
        complete = self._cluster_mass_compute_completeness(logM, z)

        return complete * intp_logM * d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_logM_intp_bin_c_d2logM_integrand(
        self, logM: float, z: float
    ):
        ccl_cosmo = self.info.ccl_cosmo
        logM_obs_lower = self.info.logM_obs_lower
        logM_obs_upper = self.info.logM_obs_upper

        return self._cluster_mean_mass_logM_intp_bin_c_d2logM(
            ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper
        )

    def _cluster_mean_mass_logM_intp_bin_c_logM(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):
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

        return logM * DeltaOmega / N

    def _cluster_mean_mass_d2logM_integrand(self, logM: float, z: float):
        ccl_cosmo = self.info.ccl_cosmo
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        return d2NdzdlogM * dvdz * logM

    def _cluster_mean_mass_bin_logM(
        self, ccl_cosmo, logM_lower, logM_upper, z_lower, z_upper
    ):
        r"""Computes the integral of $d2m(logM, z)$ over the true values of mass and redshift\
        for the bins of $logM^{\alpha}$ and $z^{\beta}$, that is  
        .. math::
            logM^{\alpha \beta} = \Omega \int_{logM^{\alpha}}^{logM^{\alpha+1}}\int_{z^\beta}^{z^\beta}\frac{d2m}{dlogMdz} \frac{dv}{dz} dlogM dz.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology

        return
        ______
        logM^{\alpha \beta}: float
            Cluster mean mass in the interval [logM^\alpha, logM^{\alpha+1}] and [z^{\beta}, z^{\beta +1 }].
        """
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
                if self.selection_error[0] == False:
                    self._compute_bin_logM = self._cluster_mean_mass_logM_intp_bin_logM
                    self._compute_intp_bin_d2logM = (
                        self._cluster_mean_mass_logM_intp_bin_d2logM
                    )
                else:
                    self._compute_bin_logM = (
                        self._cluster_mean_mass_logM_intp_bin_c_logM
                    )
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
