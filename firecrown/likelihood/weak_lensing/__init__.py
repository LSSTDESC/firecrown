"""Weak lensing source and systematics.

This subpackage provides weak lensing source classes and systematics
for use in likelihood calculations.
"""

# Re-export all public classes from private submodules
from firecrown.likelihood.weak_lensing._args import WeakLensingArgs
from firecrown.likelihood.weak_lensing._factories import (
    LinearAlignmentSystematicFactory,
    MultiplicativeShearBiasFactory,
    TattAlignmentSystematicFactory,
    WeakLensingFactory,
    WeakLensingSystematicFactory,
)
from firecrown.likelihood.weak_lensing._source import WeakLensing
from firecrown.likelihood.weak_lensing._systematics import (
    HMAlignmentSystematic,
    LinearAlignmentSystematic,
    MassDependentLinearAlignmentSystematic,
    MultiplicativeShearBias,
    PhotoZShift,
    PhotoZShiftandStretch,
    SelectField,
    TattAlignmentSystematic,
    WeakLensingSystematic,
)

# Re-export shared factories from base module
from firecrown.likelihood_base import (
    PhotoZShiftandStretchFactory,
    PhotoZShiftFactory,
)

__all__ = [
    "HMAlignmentSystematic",
    "LinearAlignmentSystematic",
    "MassDependentLinearAlignmentSystematic",
    "LinearAlignmentSystematicFactory",
    "MultiplicativeShearBias",
    "MultiplicativeShearBiasFactory",
    "PhotoZShift",
    "PhotoZShiftFactory",
    "PhotoZShiftandStretch",
    "PhotoZShiftandStretchFactory",
    "SelectField",
    "TattAlignmentSystematic",
    "TattAlignmentSystematicFactory",
    "WeakLensing",
    "WeakLensingArgs",
    "WeakLensingFactory",
    "WeakLensingSystematicFactory",
    "WeakLensingSystematic",
]
