import pyccl as ccl
import numpy as np

from ..core import Systematic

__all__ = ['LinearBiasSystematic']


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
    following Joachimi and Bridle (2010), arXiv:0911.2454 (Appendix C).

    Parameters
    ----------
    r_lim : str
        The name of the limiting magnitude in r band filter.

    Methods
    -------
    apply : appaly the systematic to a source
    """
    def __init__(self, r_lim):
        self.r_lim = r_lim

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
        b = np.array([[0.44827, 0.0, 0.0], [-1, 1, 1], [0.05617, 0.19658, 0.18107],
        [0.07704, 3.31359, 3.05213], [-11.3768, -2.5028, -2.5027]])
        a = b[0] + b[1] * np.power(b[2] * params[self.r_lim] - b[3], b[4])
        z = source.z_
        z_pow = [np.ones_like(z), z, z*z]
        pref = a @ z_pow
        source.mag_bias_ = 0.4 * pref
