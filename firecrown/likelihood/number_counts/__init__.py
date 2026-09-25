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
    # Sources and systematics
    "ConstantMagnificationBiasSystematic",
    "LinearBiasSystematic",
    "MagnificationBiasSystematic",
    "NumberCounts",
    "PhotoZShift",
    "PhotoZShiftandStretch",
    "PTNonLinearBiasSystematic",
    "SelectField",
    # Factories
    "ConstantMagnificationBiasSystematicFactory",
    "LinearBiasSystematicFactory",
    "MagnificationBiasSystematicFactory",
    "NumberCountsFactory",
    "NumberCountsSystematicFactory",
    "PhotoZShiftandStretchFactory",
    "PhotoZShiftFactory",
    "PTNonLinearBiasSystematicFactory",
]
