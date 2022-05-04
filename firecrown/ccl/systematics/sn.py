import pyccl as ccl

from ..core import Systematic

__all__ = ["AdditiveBias"]


class AdditiveBias(Systematic):
    """Additive bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.

    Parameters
    ----------
    m : str
        The name of the additive bias parameter.

    Methods
    -------
    apply : apply the systematic to a source
    """

    def __init__(self, m: str):
        """"Create a AdditiveBias with bias parameter name `m`"""
        self.m = m

    def apply(self, cosmo, params, source):
        """Apply additive bias to a source. The redshift is
        added by `(dz)`.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        source.scale_ += params[self.m]

