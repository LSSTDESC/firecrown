r"""Cluster Abundance Module
abstract class to compute cluster abundance.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional

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


class ClusterAbundance():
    r"""Cluster Abundance module.
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
    """
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

        self.funcs = self._cluster_abundance_funcs()
        self.info = None


    def _cluster_abundance_z_p_logM_p_d2n_integrand(self, logM: float, z: float):
        r"""Define integrand for the case when we have proxy for
        redshift and mass.
        The integrand is given by
        .. math::
            d2n(logM, logM_obs, z, z_obs) = \frac{d2n}{dlogMdz}  P(z_obs|logM, z)  P(logM_obs|logM, z) \frac{dv}{dz} dlogM dz.

        parameters
        __________
        logM: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        ccl_cosmo = self.info.ccl_cosmo
        z_obs = self.info.z_obs
        logM_obs = self.info.logM_obs

        p_z = self.cluster_z.cluster_redshift_p(logM, z, z_obs)
        p_logM = self.cluster_m.cluster_logM_p(logM, z, logM_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return p_z * p_logM * d2NdzdlogM * dvdz

    def cluster_abundance_z_p_logM_p_d2n(self, ccl_cosmo, logM_obs: float, z_obs: float):
        r"""Computes the integral of $d2n(logM, logM_obs, z, z_obs)$ over
        the true values of mass and redshift, that is
        .. math::
            d2n(logM_obs, z_obs) = \int_{logM_min}^{logM_max}\int_{z_min}^{z_max}\frac{d2n}{dlogMdz}  P(z_obs|logM, z)  P(logM_obs|logM, z) \frac{dv}{dz} dlogM dz.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology
        logM_obs: float
            Observed cluster mass given by log10(M) where\
            M is in units of M_sun (comoving).
        z_obs : float
            Observed cluster redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

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


    def _cluster_abundance_z_p_d2n_integrand(self, z: float):
        r"""Define integrand for the case when we have proxy\
        for redshift and true mass. The integrand is given by\
        .. math::
            d2n(logM, z, z_obs) = \frac{d2n}{dlogMdz}  P(z_obs|logM, z) \frac{dv}{dz} dz.

        parameters
        __________
        logM: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """       

        ccl_cosmo = self.info.ccl_cosmo
        logM = self.info.logM
        z_obs = self.info.z_obs


        p_z = self.cluster_z.cluster_redshift_p(logM, z, z_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return p_z * d2NdzdlogM * dvdz


    def cluster_abundance_z_p_d2n(self, ccl_cosmo, logM: float, z_obs: float):
        r"""Computes the integral of $d2n(logM, z, z_obs)$ over\
        the true values of redshift, that is
        .. math::
            d2n(logM, z_obs) = \int_{z_min}^{z_max}\frac{d2n}{dlogMdz}  P(z_obs|logM, z)  \frac{dv}{dz} dz.

        parameters
        __________

        ccl_cosmo: Cosmology
            Pyccl cosmology
        logM: float
            Cluster mass given by log10(M) where M is\
            in units of M_sun (comoving).
        z_obs : float
            Observed Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        self.info = ClusterAbundanceInfo(ccl_cosmo, logM=logM, z_obs=z_obs)

        integrand = self._cluster_abundance_z_p_d2n_integrand
        nm = scipy.integrate.quad(
                        integrand,
                        self.cluster_z.zl,
                        self.cluster_z.zu,
                    )[0]
        return nm

    def _cluster_abundance_logM_p_d2n_integrand(self, logM: float):
        r"""Define integrand for the case when we have proxy for\
        mass and true redshift.
        The integrand is given by
        .. math::
            d2n(logM, logM_obs, z) = \frac{d2n}{dlogMdz} P(logM_obs|logM, z) \frac{dv}{dz} dlogM.

        parameters
        __________
        logM: float
            Cluster mass given by log10(M) where M is in\
            units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        ccl_cosmo = self.info.ccl_cosmo
        z = self.info.z
        logM_obs = self.info.logM_obs

        p_logM = self.cluster_m.cluster_logM_p(logM, z, logM_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return p_logM * d2NdzdlogM * dvdz

    def cluster_abundance_logM_p_d2n(self, ccl_cosmo, logM_obs: float, z: float):
        r"""Computes the integral of $d2n(logM, logM_obs, z)$ over\
        the true values of mass, that is
        .. math::
            d2n(logM_obs, z) = \int_{logM_min}^{logM_max}\frac{d2n}{dlogMdz}  P(logM_obs|logM, z) \frac{dv}{dz} dlogM.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology
        logM_obs: float
            Observed cluster mass given by log10(M) where M is in\
            units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        self.info = ClusterAbundanceInfo(ccl_cosmo, z=z, logM_obs=logM_obs)

        integrand = self._cluster_abundance_logM_p_d2n_integrand
        nm = scipy.integrate.quad(
                        integrand,
                        self.cluster_m.logMl,
                        self.cluster_m.logMu,
                    )[0]
        return nm

    def _cluster_abundance_d2n_integrand(self, logM: float, z: float):
        r"""Define integrand for the case when we have\
        true redshift and mass.
        The integrand is given by
        .. math::
            d2n(logM, z) = \frac{d2n}{dlogMdz} \frac{dv}{dz}.

        parameters
        __________
        logM: float
            Cluster mass given by log10(M) where M is in\
            units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        ccl_cosmo = self.info.ccl_cosmo
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        return d2NdzdlogM * dvdz

    def _cluster_abundance_d2n(self, ccl_cosmo, logM: float, z: float):
        r"""Computes $d2n(logM, z)$ over the true values of mass, that is
        .. math::
            d2n(logM_obs, z) = \frac{d2n}{dlogMdz}  \frac{dv}{dz}.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology object.
        logM: float
            Cluster mass given by log10(M) where M is in\
            units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        self.info = ClusterAbundanceInfo(ccl_cosmo)
        return self._cluster_abundance_d2n_integrand(logM, z)

    def _cluster_abundance_z_intp_logM_intp_d2n(self, ccl_cosmo, logM: float, z: float):
        r"""Compute d2n for the case when we have proxy for redshift and mass.
        The integrand is given by
        .. math::
            d2n(logM, z) = \int_{logM_obs_min}^{logM_obs_max}\int_{z_obs_min}^{z_obs_max}\frac{d2n}{dlogMdz}  P(z_obs|logM, z)  P(logM_obs|logM, z) \frac{dv}{dz} dlogM_obs dz_obs.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology.
        logM: float
            Cluster mass given by log10(M) where M is in\
            units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """
        intp_z = self.cluster_z.cluster_redshift_intp(logM, z)
        intp_logM = self.cluster_m.cluster_logM_intp(logM, z)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * intp_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_z_intp_logM_intp_d2n_integrand(self, logM: float, z: float):

        ccl_cosmo = self.info.ccl_cosmo

        return self._cluster_abundance_z_intp_logM_intp_d2n(ccl_cosmo, logM, z)


    def _cluster_abundance_z_intp_logM_intp_N(self, ccl_cosmo):
        r"""Computes the integral of $d2n(logM, z)$ over\
        the true values of mass and redshift, that is
        .. math::
            N = \Omega \int_{logM_min}^{logM_max}\int_{z_min}^{z_max}\frac{d2n}{dlogMdz} \frac{dv}{dz} dlogM dz.

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
        N: float
            Cluster number counts in the interval [logM_lower, logM_upper], [z_lower, z_min], [logM_obs_lower, logM_obs_upper] and [z_obs_lower, zobs_min].
        """

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


    def _cluster_abundance_z_intp_d2n(self, ccl_cosmo, logM: float, z: float):
        r"""Computes d2n for the case when we have proxy for\
        redshift and true mass. The integrand is given by
        .. math::
            d2n(logM, z) = \int_{z_obs_min}^{z_obs_max}\frac{d2n}{dlogMdz}  P(z_obs|logM, z)  \frac{dv}{dz} dz_obs.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology

        logM: float
            Cluster mass given by log10(M) where M is in\
            units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        intp_z = self.cluster_z.cluster_redshift_intp(logM, z)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * d2NdzdlogM * dvdz

    def _cluster_abundance_z_intp_d2n_integrand(self, logM: float, z: float):
        ccl_cosmo = self.info.ccl_cosmo

        return self._cluster_abundance_z_intp_d2n(ccl_cosmo, logM, z)


    def _cluster_abundance_z_intp_N(self, ccl_cosmo):
        r"""Computes the integral of $d2n(logM, z)$ over\
        the true values of mass and redshift, that is
        .. math::
            N = \Omega \int_{logM_min}^{logM_max}\int_{z_min}^{z_max}\frac{d2n}{dlogMdz} \frac{dv}{dz} dlogM dz.

        In the above, we utilize the analitical integral of the redshift proxy.
        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology

        return
        ______
        N: float
            Cluster number counts in the interval [logM_lower, logM_upper],\
            [z_lower, z_min] and [z_obs_lower, zobs_min].
        """
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


    def _cluster_abundance_logM_intp_d2n(self, ccl_cosmo, logM: float, z: float):
        r"""Define integrand for the case when we have proxy\
        for mass and true redshift. The integrand is given by
        .. math::
            d2n(logM, z) = \int_{logM_obs_min}^{logM_obs_max}\frac{d2n}{dlogMdz}  P(z_obs|logM, z)  \frac{dv}{dz} dlogM_obs.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl Cosmology.
        logM: float
            Cluster mass given by log10(M) where M is in\
            units of M_sun (comoving).
        z : float
            Cluster Redshift.

        return
        ______
        d2n: float
            integrand of the counts integral.
        """

        intp_logM = self.cluster_m.cluster_logM_intp(logM, z)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_logM_intp_d2n_integrand(self, logM: float, z):
        ccl_cosmo = self.info.ccl_cosmo

        return self._cluster_abundance_logM_intp_d2n(ccl_cosmo, logM, z)

    def _cluster_abundance_logM_intp_N(self, ccl_cosmo):
        r"""Computes the integral of $d2n(logM, z)$ over\
        the true values of mass and redshift, that is
        .. math::
            N = \Omega \int_{logM_min}^{logM_max}\int_{z_min}^{z_max}\frac{d2n}{dlogMdz} \frac{dv}{dz} dlogM dz.
        
        In the above, we utilize the analitical integral of the mass proxy.
        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology

        return
        ______
        N: float
            Cluster number counts in the interval [logM_lower, logM_upper],\
            [z_lower, z_min] and [logM_obs_lower, logM_obs_upper].
        """

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


    def _cluster_abundance_N(self, ccl_cosmo):
        r"""Computes the integral of $d2n(logM, z)$ over\
        the true values of mass and redshift, that is
        .. math::
            N = \Omega \int_{logM_min}^{logM_max}\int_{z_min}^{z_max}\frac{d2n}{dlogMdz} \frac{dv}{dz} dlogM dz.

        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology


        return
        ______
        N: float
            Cluster number counts in the interval [logM_lower, logM_upper]\
            and [z_lower, z_min].
        """

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
        if self.cluster_m.use_proxy is False: 
            if self.cluster_z.use_proxy is False:
                self._compute_N = self._cluster_abundance_N
                self._compute_intp_d2n = self._cluster_abundance_d2n
            else:
                self._compute_N = self._cluster_abundance_z_intp_N
                self._compute_intp_d2n = self._cluster_abundance_z_intp_d2n
        else:
            if self.cluster_z.use_proxy is False:
                self._compute_N = self._cluster_abundance_logM_intp_N
                self._compute_intp_d2n = self._cluster_abundance_logM_intp_d2n
            else:
                self._compute_N = self._cluster_abundance_z_intp_logM_intp_N
                self._compute_intp_d2n = self._cluster_abundance_z_intp_logM_intp_d2n
        return True

    def compute_N(self, ccl_cosmo):
        return self._compute_N(ccl_cosmo)

    def compute_intp_d2n(self, ccl_cosmo, logM, z):
        return self._compute_intp_d2n(ccl_cosmo, logM, z)
