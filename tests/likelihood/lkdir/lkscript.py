"""
Provides a trivial likelihood factory function for testing purposes.
"""

from firecrown.modeling_tools import ModelingTools
from firecrown.modeling_tools import CCLFactory, PoweSpecAmplitudeParameter

# Support both relative import (when used as package) and absolute import
# (when lkdir is in sys.path)
try:
    from . import lkmodule
except ImportError:
    import lkmodule  # type: ignore


def build_likelihood(_):
    """Return an EmptyLikelihood object."""
    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    tools.test_attribute = "test"  # type: ignore[unused-ignore]
    return (lkmodule.empty_likelihood(), tools)
