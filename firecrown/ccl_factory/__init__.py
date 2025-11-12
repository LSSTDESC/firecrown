"""This module contains the CCLFactory class and it supporting classes.

The CCLFactory class is a factory class that creates instances of the
`pyccl.Cosmology` class.
"""

# Import all public types and classes from private submodules
from firecrown.ccl_factory._enums import (
    CCLCreationMode,
    CCLPureModeTransferFunction,
    PoweSpecAmplitudeParameter,
)
from firecrown.ccl_factory._factory import CCLFactory
from firecrown.ccl_factory._models import (
    CAMBExtraParams,
    CCLSplineParams,
    MuSigmaModel,
)
from firecrown.ccl_factory._types import Background, CCLCalculatorArgs, PowerSpec

# Define __all__ for explicit API contract
__all__ = [
    # Type definitions
    "PowerSpec",
    "Background",
    "CCLCalculatorArgs",
    # Enum classes
    "PoweSpecAmplitudeParameter",
    "CCLCreationMode",
    "CCLPureModeTransferFunction",
    # Model classes
    "MuSigmaModel",
    "CAMBExtraParams",
    "CCLSplineParams",
    # Factory class
    "CCLFactory",
]
