"""Deprecated: Use firecrown.modeling_tools instead.

This module is deprecated and will be removed in a future version.
All functionality has been moved to firecrown.modeling_tools.
"""

import warnings

# Re-export everything from new location
# Import must come after warnings to emit deprecation at import time
from firecrown.modeling_tools import (  # noqa: E402
    Background,
    CAMBExtraParams,
    CCLCalculatorArgs,
    CCLCreationMode,
    CCLFactory,
    CCLPureModeTransferFunction,
    CCLSplineParams,
    MuSigmaModel,
    PoweSpecAmplitudeParameter,
    PowerSpec,
)

# Emit deprecation warning when module is imported
warnings.warn(
    "firecrown.ccl_factory is deprecated and will be removed in a future version. "
    "Use firecrown.modeling_tools instead.",
    DeprecationWarning,
    stacklevel=2,
)

# pylint: disable=duplicate-code
__all__ = [
    "PowerSpec",
    "Background",
    "CCLCalculatorArgs",
    "PoweSpecAmplitudeParameter",
    "CCLCreationMode",
    "CCLPureModeTransferFunction",
    "MuSigmaModel",
    "CAMBExtraParams",
    "CCLSplineParams",
    "CCLFactory",
]
