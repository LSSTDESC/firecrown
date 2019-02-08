import functools
import numpy as np
import scipy.integrate

import pyccl as ccl
from firecrown.ccl.core import Systematic
from .kecorrections import kcorr, ecorr

# constant from KEB16, near eqn 7
C1RHOC = 0.0134

# constant from KEB16, table 1, LSST row
MLIM = 27.0

# value used to clip large luminosities
# also serves as an upper integration limit
MAX_LNL = 500  # exp(500) ~ 1e217


def _mag_to_lum(mag):
    """convert a magnitude to a luminosity

    The luminosities are always used in ratios so the zero point falls out.
    """
    # a zero point of 20 is good for numerical stability
    return np.power(10, (mag + 20) / -2.5)


def _abs_mag_lim(mlim, DLz, kz):
    # eqn 26 of KEB16
    return mlim - (5.0 * np.log10(DLz) + 25 + kz)


def _schechter_lf(L, z, phi0, Mstar, alpha, P, Q):
    """Schechter LF.

    See KEB16, eqns 21-23

    Output units are number per (Mpc/h)**3.
    """
    phiz = phi0 * np.power(10, 0.4 * P * z)
    Mstarz = Mstar - Q * (z - 0.1)
    Lstarz = _mag_to_lum(Mstarz)
    Lrat = L / Lstarz
    res = phiz * np.power(Lrat, alpha) * np.exp(-1.0 * Lrat)
    assert np.all(res >= 0), res
    return res


def _lf_red(lnL, z, phi0=1.1e-2, Mstar=-20.34, alpha=-0.57, P=-1.2, Q=1.8):
    """Red galaxy luminosity function.

    See KEB16, eqns 21-23

    This function uses the "DEEP2" parameters from table 2 of KEB16.
    """
    L = np.exp(np.clip(lnL, None, MAX_LNL))
    # factor of L at the end is for doing the integral in lnL
    return _schechter_lf(L, z, phi0, Mstar, alpha, P, Q) * L


def _lf_all(lnL, z, phi0=9.4e-3, Mstar=-20.70, alpha=-1.23, P=1.8, Q=0.7):
    """All galaxy luminosity function.

    See KEB16, eqns 21-23

    This function uses the "DEEP2" parameters from table 2 of KEB16.
    """
    L = np.exp(np.clip(lnL, None, MAX_LNL))
    # factor of L at the end is for doing the integral in lnL
    return _schechter_lf(L, z, phi0, Mstar, alpha, P, Q) * L


@functools.lru_cache(maxsize=1024)
def _compute_red_frac_z_Az(z, dlum, beta_ia, lpiv_beta_ia):
    low_lim = np.log(_mag_to_lum(_abs_mag_lim(
        MLIM,
        dlum,
        float(kcorr(z) + ecorr(z)))))
    up_lim = MAX_LNL

    # the factors below are from eqns 24 and 25 of KEB16
    red_intg = scipy.integrate.quad(
        _lf_red,
        low_lim,
        up_lim,
        args=(z,))

    all_intg = scipy.integrate.quad(
        _lf_all,
        low_lim,
        up_lim,
        args=(z,))

    def _func(lnL):
        L = np.exp(np.clip(lnL, None, MAX_LNL))
        # factor of L at the end is for doing the integral in lnL
        return _lf_red(lnL, z) * np.power(L / lpiv_beta_ia, beta_ia) * L

    red_wgt_intg = scipy.integrate.quad(
        _func,
        low_lim,
        up_lim)

    return red_intg[0] / all_intg[0], red_wgt_intg[0] / red_intg[0]


class KEBNLASystematic(Systematic):
    """KEB NLA systematic.

    This systematic adds the KEB non-linear, linear alignment (NLA) intrinsic
    alignment model which varies with redshift, luminosity, and
    the growth function.

    Parameters
    ----------
    eta_ia : str
        The mame of redshift dependence parameter of the intrinsic alignment
        signal.
    eta_ia_highz : str
        The mame of redshift dependence parameter of the high-z intrinsic
        alignment signal.
    beta_ia : str
        The name of the power-law parameter for the luminosity dependence of
        the intrinsic alignment signal.
    Omega_b : str
        The name of the parameter for the baryon density at z = 0.
    Omega_c : str
        The name of the patameter for the cold dark matter density at z = 0.

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(self, eta_ia, eta_ia_highz, beta_ia, Omega_b, Omega_c):
        self.eta_ia = eta_ia
        self.eta_ia_highz = eta_ia_highz
        self.beta_ia = beta_ia
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c

        # set internal **constants**
        self._zpiv_eta_ia = 0.3
        self._zpiv_eta_ia_highz = 0.7
        self._lpiv_beta_ia = _mag_to_lum(-22)

    def apply(self, cosmo, params, source):
        """Apply a linear alignment systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        red_frac = []
        ia_bias = []
        for z in source.z_:
            dlum = ccl.luminosity_distance(cosmo, 1 / (1.0 + z))
            rf, az = _compute_red_frac_z_Az(
                z, dlum, params[self.beta_ia], self._lpiv_beta_ia)
            red_frac.append(rf)

            # eqn 7 of KEB16 without A0 (already in ia_bias)
            az_low = (
                az *
                (params[self.Omega_b] + params[self.Omega_c]) *
                C1RHOC /
                ccl.growth_factor(cosmo, 1.0 / (1.0 + z)) *
                np.power((1 + z) / (1 + self._zpiv_eta_ia),
                         params[self.eta_ia]))

            # eqn 8 of KEB16
            az = az_low * (
                1.0 * (z < self._zpiv_eta_ia_highz) +
                np.power((1 + z) / (1 + self._zpiv_eta_ia_highz),
                         params[self.eta_ia_highz]) * (
                            z > self._zpiv_eta_ia_highz))

            ia_bias.append(az)

        source.ia_bias_ *= np.array(ia_bias)
        source.red_frac_ *= np.array(red_frac)


class DESCSRDv1MultiplicativeShearBias(Systematic):
    """DESC SRD v1 Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`
    where `m` depends on redshift according to `(2z - zmax) / (zmax)`.

    In this version we average `m` over the source dndz. This procedure
    is not exactly correct, but the SRD document didn't say what they
    did either, so shrug.

    Parameters
    ----------
    m : str
        The name of the multiplicative bias parameter.

    Methods
    -------
    apply : apply the systematic to a source
    """
    def __init__(self, m):
        self.m = m
        self._zmax = 1.33  # set in stone now and forever

    def apply(self, cosmo, params, source):
        """Apply multiplicative shear bias to a source. The `scale_` of the
        source is multiplied by `(1 + m)` where `m` depends on redshift
        according to `(2z - zmax) / (zmax)`.

        In this version we average `m` over the source dndz. This procedure
        is not exactly correct, but the SRD document didn't say what they
        did either, so shrug.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        nrm = np.sum(source.dndz_)
        zfac = np.sum(
            (2.0 * source.z_ - self._zmax) / self._zmax * source.dndz_) / nrm
        source.scale_ *= (1.0 + params[self.m] * zfac)
