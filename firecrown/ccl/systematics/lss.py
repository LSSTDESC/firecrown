import pyccl as ccl
import numpy as np

from ..core import Systematic

__all__ = ['LinearBiasSystematic', 'MagnificationBias']


class LinearBiasSystematic(Systematic):
    """Linear alignment systematic.

    This systematic adds a linear bias model which varies with redshift and
    the growth function.

    Parameters
    ----------
    alphaz : str
        The mame of redshift dependence parameter of the linear bias.
    alphag : str
        The name of the growth dependence parameter of the linear bias.
    z_piv : str
        The name of the pivot redshift parameter for the linear bias.
    Methods
    -------
    apply : appaly the systematic to a source
    """
    def __init__(self, alphaz, alphag, z_piv):
        self.alphaz = alphaz
        self.alphag = alphag
        self.z_piv = z_piv

    def apply(self, cosmo, params, source):
        """Apply a linear bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        pref = (
            ((1.0 + source.z_) / (1.0 + params[self.z_piv])) **
            params[self.alphaz])
        pref *= ccl.growth_factor(
                cosmo, 1.0 / (1.0 + source.z_)) ** params[self.alphag]
        source.bias_ *= pref


class MagnificationBias(Systematic):
    """Magnification bias systematic.

    This systematic adds a magnification bias model for galaxy number contrast
    following Joachimi & Bridle (2010), arXiv:0911.2454.

    Parameters
    ----------
    r_lim : str
        The name of the limiting magnitude in r band filter.
    Sig_c, eta, z_c, z_m : str
        The name of the fitting parameters in Joachimi & Bridle (2010) equation
    (C.1).

    Methods
    -------
    apply : appaly the systematic to a source
    """
    def __init__(self, r_lim, Sig_c, eta, z_c, z_m):
        self.r_lim = r_lim
        self.Sig_c = Sig_c
        self.eta = eta
        self.z_c = z_c
        self.z_m = z_m

    def apply(self, cosmo, params, source):
        """Apply a magnification bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """

        z_bar = params[self.z_c] + params[self.z_m] * (params[self.r_lim] - 24)
        z = source.z_
        s = (
            params[self.eta] / params[self.r_lim] - 3 * params[self.z_m] /
            z_bar + 1.5 * params[self.z_m] * np.power(z / z_bar, 1.5) / z_bar)
        source.mag_bias_ = s / np.log(10)
