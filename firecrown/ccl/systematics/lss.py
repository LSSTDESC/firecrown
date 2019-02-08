import pyccl as ccl

from ..core import Systematic

__all__ = ['LinearBiasSystematic']


class LinearBiasSystematic(Systematic):
    """Linear bias systematic.

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
    apply : apply the systematic to a source
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
