"""
Provides a trivial likelihood factory function for testing purposes.
The likelihood created provides one derived parameter named "derived_param0".
"""

from firecrown.likelihood.likelihood import NamedParameters
from . import lkmodule


def build_likelihood(_: NamedParameters):
    """Return a DerivedParameterLikelihood object."""
    return lkmodule.derived_parameter_likelihood()
