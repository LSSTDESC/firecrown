r"""Cluster Abundance Module
abstract class to compute cluster abundance.
========================================
The implemented functions use PyCCL library as backend.
"""
from __future__ import annotations
from typing import List, Optional, Union

import numpy as np
import scipy.integrate


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
        sky_area: Optional[float] = None,
        selection_error=[False, False]
        ):
        self.cluster_m = cluster_mass
        self.cluster_z = cluster_redshift
        self.sky_area = sky_area
        self.selection_error = selection_error
        self.compute_counts_integrand = None
        self.compute_d2n = None
        self.compute_intp_d2n = None
        self.compute_intp_N = None
        self.compute_N = None

        self.funcs = self._cluster_abundance_funcs()


#Selection Function Errors
    def _cluster_abundance_compute_purity(self, logM_obs, z):
        ln_r = np.log(10**logM_obs)
        a_nc = np.log(10) * 0.8612
        b_nc = np.log(10) * 0.3527
        a_rc = 2.2183
        b_rc = -0.6592
        nc = a_nc + b_nc * (1.0 + z)
        ln_rc = a_rc + b_rc * (1.0 + z)
        purity = (ln_r / ln_rc) ** nc / ((ln_r / ln_rc) ** nc + 1.0)
        return purity

    def _cluster_abundance_compute_completeness(self, logM, z):
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc * (1.0 + z)
        nc = a_nc + b_nc * (1.0 + z)
        C = (logM / log_mc) ** nc / ((logM / log_mc) ** nc + 1.0)
        return C

#d2n functions. Either the integrand, which means that d2n*p*v depends on all the
#variables, or \int d2n*p*v that depends always on two variables, either the proxies
#or the true ones.

    def _cluster_abundance_z_p_logM_p_counts_integrand(self, ccl_cosmo, logM: float, z: float, logM_obs, z_obs):
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

        p_z = self.cluster_z.cluster_redshift_p(logM, z, z_obs)
        p_logM = self.cluster_m.cluster_logM_p(logM, z, logM_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return p_z * p_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_z_p_logM_p_d2n(self, ccl_cosmo, logM_obs: float, z_obs: float):
        r"""Computes the integral of $d2n(logM, logM_obs, z, z_obs)$ over
        the true values of mass and redshift, that is
        .. math::
            d2n(logM_obs, z_obs) = \int_{logM_lower}^{logM_upper}\int_{z_lower}^{z_upper}\frac{d2n}{dlogMdz}  P(z_obs|logM, z)  P(logM_obs|logM, z) \frac{dv}{dz} dlogM dz.

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
        def integrand(logM, z):
            return self._cluster_abundance_z_p_logM_p_counts_integrand(ccl_cosmo, logM, z, logM_obs, z_obs)
        nm = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_redshift.zl,
                        self.cluster_redshift.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4
                    )[0]
        return nm


    def _cluster_abundance_z_p_counts_integrand(self, ccl_cosmo, z: float, logM, z_obs):
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

        p_z = self.cluster_z.cluster_redshift_p(logM, z, z_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo,logM,z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return p_z * d2NdzdlogM * dvdz


    def _cluster_abundance_z_p_d2n(self, ccl_cosmo, logM: float, z_obs: float):
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
        def integrand(z):
            return self._cluster_abundance_z_p_counts_integrand(ccl_cosmo, z, logM_obs, z_obs)
        nm = scipy.integrate.quad(
                        integrand,
                        self.cluster_redshift.zl,
                        self.cluster_redshift.zu,
                    )[0]
        return nm

    def _cluster_abundance_logM_p_counts_integrand(self,  ccl_cosmo, logM: float, z, logM_obs):
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

        p_logM = self.cluster_m.cluster_logM_p(logM, z, logM_obs)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return p_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_logM_p_d2n(self, ccl_cosmo, logM_obs: float, z: float):
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
        def integrand(logM):
            return self._cluster_abundance_logM_p_counts_integrand(ccl_cosmo, logM, z, logM_obs)
        nm = scipy.integrate.quad(
                        integrand,
                        self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        self.cluster_m.logMu
                    )[0]

        return nm


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

        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        return d2NdzdlogM * dvdz

#Counts function that uses one of the above functions
#THis function is useless honestly. It is so costly to call a integrand that call another integral.
# THis has to be a triple integral if we have one proxy or a 4d integral if we have both proxies

    def _cluster_abundance_N(self, ccl_cosmo, logM_lower: float, logM_upper: float, z_lower: float, z_upper: float):
        r"""Computes the integral of $d2n(logM, z)$ over\
        the values of mass and redshift, that is
        .. math::
            N = \Omega \int_{logM_min}^{logM_max}\int_{z_min}^{z_max}\frac{d2n}{dlogMdz} \frac{dv}{dz} dlogM dz.
        Note that if the user pass proxies for the mass and the redshift, this integral
        is over the oberved parameters and not the true ones.
        parameters
        __________
        ccl_cosmo: Cosmology
            Pyccl cosmology


        return
        ______
        N: float
            Cluster number counts in the interval [logM_lower, logM_upper]\
            and [z_lower, z_upper].
        """
        def integrand(logM, z):
            return self.compute_d2n(ccl_cosmo, logM, z)

        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        z_lower,
                        z_upper,
                        # pylint: disable-next=cell-var-from-loop
                        logM_lower,
                        # pylint: disable-next=cell-var-from-loop
                        logM_upper,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega


#Now the same d2n functions but for the analitical proxy, i.e., integrated
# on the M_obs or z_obs and their respective counts function.

    def _cluster_abundance_z_intp_logM_intp_d2n(self, ccl_cosmo, logM: float, z: float, logM_obs_lower: float, logM_obs_upper: float, z_obs_lower: float, z_obs_upper: float):
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
        intp_z = self.cluster_m.cluster_logM_intp(logM, z, logM_obs_lower, logM_obs_upper)
        intp_logM = self.cluster_z.cluster_redshift_intp(logM, z, z_obs_lower, z_obs_upper )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * intp_logM * d2NdzdlogM * dvdz


    def _cluster_abundance_z_intp_logM_intp_N(self, ccl_cosmo, logM_obs_lower: float, logM_obs_upper: float, z_obs_lower: float, z_obs_upper: float):
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

        def integrand(logM: float, z: float):
            return self._cluster_abundance_z_intp_logM_intp_d2n(ccl_cosmo, logM, z, logM_obs_lower,logM_obs_upper, z_obs_lower, z_obs_upper)

        DeltaOmega = self.sky_area * np.pi**2 / 180**2
        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_redshift.zl,
                        self.cluster_redshift.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMl,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: self.cluster_m.logMu,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]
        return N * DeltaOmega


    def _cluster_abundance_z_intp_d2n(self, ccl_cosmo, logM: float, z: float, z_obs_lower: float, z_obs_upper: float):
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

        intp_z = self.cluster_z.cluster_redshift_intp(logM, z, z_obs_lower, z_obs_upper)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_z * d2NdzdlogM * dvdz

    def _cluster_abundance_z_intp_N(self, ccl_cosmo, logM_lower: float, logM_upper: float, z_obs_lower: float, z_obs_upper: float):
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
        def integrand(logM: float, z: float):
            return self._cluster_abundance_z_intp_d2n(ccl_cosmo, logM, z, z_obs_lower, z_obs_upper)

        DeltaOmega = self.sky_area * np.pi**2 / 180**2

        N = scipy.integrate.dblquad(
                        integrand,
                        self.cluster_redshift.zl,
                        self.cluster_redshift.zu,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: logM_lower,
                        # pylint: disable-next=cell-var-from-loop
                        lambda x: logM_upper,
                        epsabs=1.0e-4,
                        epsrel=1.0e-4,
                    )[0]

        return N * DeltaOmega


    def _cluster_abundance_logM_intp_d2n(self, ccl_cosmo, logM: float, z: float, logM_obs_lower: float, logM_obs_upper: float):
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

        intp_logM = self.cluster_m.cluster_logM_intp(logM, z, logM_obs_lower, logM_obs_upper)
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)

        return intp_logM * d2NdzdlogM * dvdz


    def _cluster_abundance_logM_intp_N(self, ccl_cosmo,  logM_obs_lower: float, logM_obs_upper: float, z_lower: float, z_upper: float):
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
        def integrand(logM: float, z):
            return self._cluster_abundance_logM_intp_d2n(ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper)


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




## Function with selection function error. THere is not all the possible cases yet. I just implemented the ones that I wanted to use
    def _cluster_abundance_logM_intp_c_d2n(
        self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper
    ):
        intp_logM = self.cluster_m.cluster_logM_intp(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        complete = self._cluster_abundance_compute_completeness(logM, z)

        return complete * intp_logM * d2NdzdlogM * dvdz

    def _cluster_abundance_logM_intp_c_N(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):

        def integrand(logM, z):
            return self._cluster_abundance_logM_intp_c_d2n(ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper)

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

    def _cluster_abundance_logM_intp_cp_d2n(
        self, ccl_cosmo, logM: float, z: float, logM_obs_lower, logM_obs_upper
    ):
        intp_logM = self.cluster_m.cluster_logM_intp(
            logM, z, logM_obs_lower, logM_obs_upper
        )
        d2NdzdlogM = self.cluster_m.compute_mass_function(ccl_cosmo, logM, z)
        dvdz = self.cluster_z.compute_differential_comoving_volume(ccl_cosmo, z)
        purity = self._cluster_abundance_compute_purity(
            (logM_obs_upper - logM_obs_lower) / 2.0, z
        )
        complete = self._cluster_abundance_compute_completeness(logM, z)
        return complete / purity * intp_logM * d2NdzdlogM * dvdz


    def _cluster_abundance_logM_intp_cp_N(
        self, ccl_cosmo, logM_obs_lower, logM_obs_upper, z_lower, z_upper
    ):
        def integrand(logM, z):
            return self._cluster_abundance_logM_intp_cp_d2n(ccl_cosmo, logM, z, logM_obs_lower, logM_obs_upper)

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


# Automatic Functions to be used

##FIrst the counts_integrand function
    def cluster_abundance_integrand(self, ccl_cosmo, logM, z, logM_obs: Optional[float] = None, z_obs: Optional[float] = None):
        counts_integrand = None

        if logM_obs is None:
            if z_obs is None:
                 counts_integrand = self._cluster_abundance_d2n(ccl_cosmo, logM, z)
            else:
                counts_integrand = self._cluster_abundance_z_p_counts_integrand(ccl_cosmo, logM, z, z_obs)
        else:
            if z_obs is None:
                counts_integrand = self._cluster_abundance_logM_p_counts_integrand(ccl_cosmo, logM, z, logM_obs)
            else:
                counts_integrand = self._cluster_abundance_z_p_logM_p_counts_integrand(ccl_cosmo, logM, z, logM_obs, z_obs)

##Function to give either the 2d Counts integral or d2n(M_obs, z_obs). It can be true or proxy variables.
##This has the same approach as the other function I said is useless.
## It is automatic but will call integral inside integral

    def cluster_abundance(self, ccl_cosmo, logM: Union[float, List], z: Union[float, List]):
        N = None
        if type(logM).__name__ == 'list' and type(z).__name__ == 'list':
            def integrand(logM_int, z_int):
                return self._compute_d2n(ccl_cosmo, logM_int, z_int)

            DeltaOmega = self.sky_area * np.pi**2 / 180**2
            N = scipy.integrate.dblquad(
                            integrand,
                            z[0],
                            z[1],
                            # pylint: disable-next=cell-var-from-loop
                            logM[0],
                            # pylint: disable-next=cell-var-from-loop
                            logM[1],
                            epsabs=1.0e-4,
                            epsrel=1.0e-4,
                        )[0]

        elif type(logM).__name__ == 'float' and type(z).__name__ == 'float':
            N = self._compute_d2n(ccl_cosmo, logM, z)

        else:
            raise ValueError(f"Input types ({type(logM).__name__ },{type(z).__name__}) for mass  and redshift are not compatible")

        return N * DeltaOmega


##Function to give either the 2d Counts integral for both proxies or d2n(M_obs, z_obs). This use the analitical integral of the proxy so it works well
    def cluster_abundance_z_intp_logM_intp(self, ccl_cosmo, logM: Union[float, List], z: Union[float, List], logM_obs: list, z_obs: list):
        N = None
        if type(logM).__name__ == 'list' and type(z).__name__ == 'list':
            def integrand(logM_int, z_int):
                return self._cluster_abundance_z_intp_logM_intp_d2n(ccl_cosmo, logM_int, z_int, logM_obs[0], logM_obs[1], z_obs[0], z_obs[1])

            DeltaOmega = self.sky_area * np.pi**2 / 180**2
            N = scipy.integrate.dblquad(
                            integrand,
                            z[0],
                            z[1],
                            # pylint: disable-next=cell-var-from-loop
                            logM[0],
                            # pylint: disable-next=cell-var-from-loop
                            logM[1],
                            epsabs=1.0e-4,
                            epsrel=1.0e-4,
                        )[0]
        elif type(logM).__name__ == 'float' and type(z).__name__ == 'float':
            N = self._cluster_abundance_z_intp_logM_intp_d2n(ccl_cosmo, logM, z, logM_obs[0], logM_obs[1], z_obs[0], z_obs[1])

        else:
            raise ValueError(f"Input types ({type(logM).__name__ },{type(z).__name__}) for mass  and redshift are not compatible")

        return N * DeltaOmega


##Function to give either the 2d Counts integral for redshift proxy or d2n(M, z_obs). This use the analitical integral of the proxy so it works well
    def cluster_abundance_z_intp(self, ccl_cosmo, logM: Union[float, List], z: Union[float, List], z_obs: list):
        N = None
        if type(logM).__name__ == 'list' and type(z).__name__ == 'list':
            def integrand(logM_int, z_int):
                return self._cluster_abundance_z_intp_d2n(ccl_cosmo, logM_int, z_int, z_obs[0], z_obs[1])

            DeltaOmega = self.sky_area * np.pi**2 / 180**2
            N = scipy.integrate.dblquad(
                            integrand,
                            z[0],
                            z[1],
                            # pylint: disable-next=cell-var-from-loop
                            logM[0],
                            # pylint: disable-next=cell-var-from-loop
                            logM[1],
                            epsabs=1.0e-4,
                            epsrel=1.0e-4,
                        )[0]
        elif type(logM).__name__ == 'float' and type(z).__name__ == 'float':
            N = self._cluster_abundance_z_intp_d2n(ccl_cosmo, logM, z, z_obs[0], z_obs[1])

        else:
            raise ValueError(f"Input types ({type(logM).__name__ },{type(z).__name__}) for mass  and redshift are not compatible")

        return N * DeltaOmega


##Function to give either the 2d Counts integral for mass proxy or d2n(M_obs, z). This use the analitical integral of the proxy so it works well
##The selection funciton option is used here
    def cluster_abundance_logM_intp(self, ccl_cosmo, logM: Union[float, List], z: Union[float, List], logM_obs: list):
        N = None
        if type(logM).__name__ == 'list' and type(z).__name__ == 'list':
            if self.selection_error == [False, False]:
                def integrand(logM_int, z_int):
                    return self._cluster_abundance_logM_intp_d2n(ccl_cosmo, logM_int, z_int, logM_obs[0], logM_obs[1])
            elif self.selection_error == [True, False]:
                def integrand(logM_int, z_int):
                    return self._cluster_abundance_logM_intp_c_d2n(ccl_cosmo, logM_int, z_int, logM_obs[0], logM_obs[1])
            elif self.selection_error == [True, True]:
                def integrand(logM_int, z_int):
                    return self._cluster_abundance_logM_intp_cp_d2n(ccl_cosmo, logM_int, z_int, logM_obs[0], logM_obs[1])

            DeltaOmega = self.sky_area * np.pi**2 / 180**2
            N = scipy.integrate.dblquad(
                            integrand,
                            z[0],
                            z[1],
                            # pylint: disable-next=cell-var-from-loop
                            logM[0],
                            # pylint: disable-next=cell-var-from-loop
                            logM[1],
                            epsabs=1.0e-4,
                            epsrel=1.0e-4,
                        )[0]
        elif type(logM).__name__ == 'float' and type(z).__name__ == 'float':
            N = self._cluster_abundance_logM_intp_d2n(ccl_cosmo, logM, z, logM_obs[0], logM_obs[1])

        else:
            raise ValueError(f"Input types ({type(logM).__name__ },{type(z).__name__}) for mass  and redshift are not compatible")

        return N * DeltaOmega


##Definition of how to choose the functions defined in what was passed to the object
##This shall be changed

    def _cluster_abundance_funcs(self):
        if self.cluster_m.use_proxy is True:
            if self.cluster_z.use_proxy is True:
                self.compute_counts_integrand = self._cluster_abundance_z_p_logM_p_counts_integrand
                self.compute_d2n = self._cluster_abundance_z_p_logM_p_d2n
                self.compute_intp_d2n = self._cluster_abundance_z_intp_logM_intp_d2n
                self.compute_intp_N = self._cluster_abundance_z_intp_logM_intp_N
                self.compute_N = self._cluster_abundance_N
            else:
                if self.selection_error[0] == False:
                    self.compute_counts_integrand = self._cluster_abundance_logM_p_counts_integrand
                    self.compute_d2n = self._cluster_abundance_logM_p_d2n
                    self.compute_intp_d2n = self._cluster_abundance_logM_intp_d2n
                    self.compute_intp_N = self._cluster_abundance_logM_intp_N
                    self.compute_N = self._cluster_abundance_N
                elif self.selection_error[0] == True and self.selection_error[1] == False:
                    self.compute_counts_integrand = self._cluster_abundance_logM_p_counts_integrand
                    self.compute_d2n = self._cluster_abundance_logM_p_d2n
                    self.compute_intp_d2n = self._cluster_abundance_logM_intp_c_d2n
                    self.compute_intp_N = self._cluster_abundance_logM_intp_c_N
                    self.compute_N = self._cluster_abundance_N
                else:
                    self.compute_counts_integrand = self._cluster_abundance_logM_p_counts_integrand
                    self.compute_d2n = self._cluster_abundance_logM_p_d2n
                    self.compute_intp_d2n = self._cluster_abundance_logM_intp_cp_d2n
                    self.compute_intp_N = self._cluster_abundance_logM_intp_cp_N
                    self.compute_N = self._cluster_abundance_N
        else:
            if self.cluster_z.use_proxy is True:
                self.compute_counts_integrand = self._cluster_abundance_z_p_counts_integrand
                self.compute_d2n = self._cluster_abundance_z_p_d2n
                self.compute_intp_d2n = self._cluster_abundance_z_intp_d2n
                self.compute_intp_N = self._cluster_abundance_z_intp_N
                self.compute_N = self._cluster_abundance_N
            else:
                self.compute_d2n = self._cluster_abundance_d2n
                self.compute_N = self._cluster_abundance_N
        return True
