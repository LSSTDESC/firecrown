"""
Provides a trivial likelihood factory function with PURE_CCL_MODE for testing purposes.
"""

from firecrown.likelihood._likelihood import NamedParameters
from firecrown.modeling_tools import (
    CCLCreationMode,
    CCLFactory,
    ModelingTools,
    PoweSpecAmplitudeParameter,
)

from . import lkmodule


def build_likelihood(
    _: NamedParameters,
) -> tuple[lkmodule.EmptyLikelihood, ModelingTools]:
    """Return an EmptyLikelihood object with PURE_CCL_MODE."""
    tools = ModelingTools(
        ccl_factory=CCLFactory(
            amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8,
            creation_mode=CCLCreationMode.PURE_CCL_MODE,
        )
    )
    tools.test_attribute = "test"  # type: ignore[unused-ignore]
    return (lkmodule.empty_likelihood(), tools)
