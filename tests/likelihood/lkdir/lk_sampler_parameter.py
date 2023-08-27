"""
Provides a trivial likelihood factory function for testing purposes.
The likelihood created requires a string parameter named "sacc_tracer"
and has a sampler parameter named "sampler_param0".
"""
from firecrown.likelihood.likelihood import NamedParameters
from . import lkmodule


def build_likelihood(params: NamedParameters):
    """Return a SamplerParameterLikelihood object."""
    return lkmodule.sampler_parameter_likelihood(params)
