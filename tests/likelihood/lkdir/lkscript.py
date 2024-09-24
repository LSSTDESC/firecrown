"""
Provides a trivial likelihood factory function for testing purposes.
"""

from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory, PoweSpecAmplitudeParameter
from . import lkmodule


def build_likelihood(_):
    """Return an EmptyLikelihood object."""
    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    tools.test_attribute = "test"  # type: ignore[unused-ignore]
    return (lkmodule.empty_likelihood(), tools)
