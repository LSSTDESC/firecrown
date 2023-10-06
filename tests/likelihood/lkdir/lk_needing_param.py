"""
Provides a trivial likelihood factory function for testing purposes.
The likelihood created requires a string parameter named "sacc_file".
"""
from firecrown.likelihood.likelihood import NamedParameters
from . import lkmodule


def build_likelihood(params: NamedParameters):
    """Return a ParameterizedLikelihood object."""
    return lkmodule.parameterized_likelihood(params)
