"""Number counts source and systematics."""

from firecrown.likelihood.number_counts._factories import (
    ConstantMagnificationBiasSystematicFactory,
    LinearBiasSystematicFactory,
    MagnificationBiasSystematicFactory,
    NumberCountsFactory,
    NumberCountsSystematicFactory,
    PTNonLinearBiasSystematicFactory,
)
from firecrown.likelihood.number_counts._source import NumberCounts
from firecrown.likelihood.number_counts._systematics import (
    ConstantMagnificationBiasSystematic,
    LinearBiasSystematic,
    MagnificationBiasSystematic,
    PhotoZShift,
    PhotoZShiftandStretch,
    PTNonLinearBiasSystematic,
    SelectField,
)

# Re-export shared factories from base module
from firecrown.likelihood_base import (
    PhotoZShiftandStretchFactory,
    PhotoZShiftFactory,
)

__all__ = [
    "NumberCounts",
    "NumberCountsFactory",
    "NumberCountsSystematicFactory",
    "PhotoZShift",
    "PhotoZShiftFactory",
    "PhotoZShiftandStretch",
    "PhotoZShiftandStretchFactory",
    "SelectField",
    "ConstantMagnificationBiasSystematic",
    "ConstantMagnificationBiasSystematicFactory",
    "LinearBiasSystematic",
    "LinearBiasSystematicFactory",
    "PTNonLinearBiasSystematic",
    "PTNonLinearBiasSystematicFactory",
    "MagnificationBiasSystematic",
    "MagnificationBiasSystematicFactory",
]
