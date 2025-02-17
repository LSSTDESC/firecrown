"""
Provides a trivial likelihood factory function for testing purposes.
The likelihood created provides one derived parameter named "derived_param0".
"""

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory, PoweSpecAmplitudeParameter
from . import lkmodule


def build_likelihood(_: NamedParameters):
    """Return a DerivedParameterLikelihood object."""
    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    return lkmodule.derived_parameter_likelihood(), tools
