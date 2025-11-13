"""Weak lensing source and systematics.

This subpackage provides weak lensing source classes and systematics
for use in likelihood calculations.
"""

# Re-export all public classes from the private module
from firecrown.likelihood._weak_lensing import (
    HMAlignmentSystematic,
    LinearAlignmentSystematic,
    LinearAlignmentSystematicFactory,
    MultiplicativeShearBias,
    MultiplicativeShearBiasFactory,
    PhotoZShift,
    PhotoZShiftandStretch,
    SelectField,
    TattAlignmentSystematic,
    TattAlignmentSystematicFactory,
    WeakLensing,
    WeakLensingArgs,
    WeakLensingFactory,
    WeakLensingSystematic,
)

__all__ = [
    "HMAlignmentSystematic",
    "LinearAlignmentSystematic",
    "LinearAlignmentSystematicFactory",
    "MultiplicativeShearBias",
    "MultiplicativeShearBiasFactory",
    "PhotoZShift",
    "PhotoZShiftandStretch",
    "SelectField",
    "TattAlignmentSystematic",
    "TattAlignmentSystematicFactory",
    "WeakLensing",
    "WeakLensingArgs",
    "WeakLensingFactory",
    "WeakLensingSystematic",
]
