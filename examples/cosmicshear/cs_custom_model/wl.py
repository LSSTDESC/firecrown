"""This is a template for you to implement

We are going to build a simple systematic which models the multiplicative
bias of each tomo bin as

    m_i = m_delta_i + m_base

where m_base is shared between the bins. This model is dumb, but YOLO.
"""
from firecrown.ccl.core import Systematic


class CustomShearBias(Systematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m_delta + m_base)`.

    Parameters
    ----------
    m_delta : str
        The name of the multiplicative bias delta parameter.
    m_base : str
        The name of the base multiplicative bias parameter.

    Methods
    -------
    apply : appaly the systematic to a source
    """
    def __init__(self, m_delta, m_base):
        # don't do any work in this function
        # you need to attach the params to the object

    def apply(self, cosmo, params, source):
        """Apply multiplicative shear bias to a source. The `scale_` of the
        source is multiplied by `(1 + m)`.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        # each parameter to a systematic is a string that references a
        # parameter in the params dict

        # you need to adjust the scale of the input source
