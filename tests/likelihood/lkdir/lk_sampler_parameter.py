"""
Provides a trivial likelihood factory function for testing purposes.
The likelihood created requires a string parameter named "parameter_prefix"
and has a sampler parameter named "sampler_param0".
"""

from firecrown.likelihood._likelihood import NamedParameters
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory, PoweSpecAmplitudeParameter
from . import lkmodule


def build_likelihood(params: NamedParameters):
    """Return a SamplerParameterLikelihood object."""
    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    return lkmodule.sampler_parameter_likelihood(params), tools
