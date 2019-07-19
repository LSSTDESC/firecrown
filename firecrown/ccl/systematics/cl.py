import pyccl as ccl
import numpy as np
import scipy.special

from ..core import Systematic

__all__ = ['MORTrue', 'MORMurata']

class MORTrue(Systematic):
    """Mass-Observable relation systematic.

    This systematic simply returns the input mass.

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(self,):
        pass

    def integrate_p_dproxy(self, params, ln_m, z, lambda_min, lambda_max):
        """Just returns 1
        """
        return 1

    def apply(self, cosmo, params, source):
        """Apply a linear bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the MOR model.
        """
        source.bias_ *= source.integrate_pmor_dz_dm_dproxy(
            cosmo, params, self, weight=ccl.halo_bias)

class MORMurata(Systematic):
    """Mass-Observable relation systematic.

    This systematic implements Murata et al. 2018 (1707.01907)
    model ln lamba = mor_a + mor_b ln(M/M_pivot) + mor_c ln(1+z)
    with mass-dependent mass-observable scatter (as used in the SRD).

    Parameters
    ----------
    mor_a : str
        The name of the MOR normalization parameter.
    mor_b : str
        The name of the MOR mass scaling parameter.
    mor_c : str
        The name of the MOR redshift scaling parameter.
    mor_scatter_s0 : str
        The name of the MOR scatter normalization parameter.
    mor_scatter_qm : str
        The name of the MOR scatter mass scaling parameter.
    mor_scatter_qz : str
        The name of the MOR scatter redshift scaling parameter.

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(
            self, *, mor_a, mor_b, mor_c, mor_scatter_s0, mor_scatter_qm,
            mor_scatter_qz):
        self.mor_a = mor_a
        self.mor_b = mor_b
        self.mor_c = mor_c
        self.mor_scatter_s0 = mor_scatter_s0
        self.mor_scatter_qm = mor_scatter_qm
        self.mor_scatter_qz = mor_scatter_qz
        _h_planck_2015 = 0.678
        # pivot mass of 1707.01907 in Msun
        self.ln_m_pivot = np.log(3.e+14*_h_planck_2015)

    def integrate_p_dproxy(self, params, ln_m, z, lambda_min, lambda_max):
        """Integral of P(proxver [proxy_min, proxy_max]
        for log-normal scatter, this is given by error functions
        c.f. Eq.18 in 1707.01907
        """

        def _mean_lnproxy_given_lnm(self, ln_m, z):
            """Mean ln(lambda)(M,z), which extends Eq.15 of 1707.01907
            with power law scaling in (1+z); c.f. Eq.7 of 1809.01669(SRD)
            """
            ln_lambda = (
                params[self.mor_a] +
                params[self.mor_b] * (ln_m - self.ln_m_pivot) +
                params[self.mor_c] * np.log(1+z))
            return ln_lambda

        def _sigma_lnproxy_given_lnm(self, ln_m, z):
            """ Scatter in ln(lambda) at fixed M,z, which extends Eq.16 of 1707.01907
            with power law scaling in (1+z); c.f. Eq.8 of 1809.01669(SRD)
            """
            sigma_ln_lambda = (
                params[self.mor_scatter_s0] +
                params[self.mor_scatter_qm] * (ln_m - self.ln_m_pivot) +
                params[self.mor_scatter_qz] * np.log(1+z))
            return sigma_ln_lambda

        _xmin = np.log(lambda_min) - _mean_lnproxy_given_lnm(ln_m, z)
        _xmax = np.log(lambda_max) - _mean_lnproxy_given_lnm(ln_m, z)
        _sigma = _sigma_lnproxy_given_lnm(ln_m, z)
        s_lnm = 0.5 * (
            scipy.special.erf(_xmax/(np.sqrt(2.)*_sigma)) -
            scipy.special.erf(_xmin/(np.sqrt(2.)*_sigma)))
        return s_lnm

    def apply(self, cosmo, params, source):
        """Apply a linear bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the MOR model.
        """
        source.bias_ *= source.integrate_pmor_dz_dm_dproxy(
            cosmo, params, self, weight=ccl.halo_bias)
