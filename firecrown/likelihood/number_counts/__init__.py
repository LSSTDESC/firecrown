"""Number counts source and systematics."""

from firecrown.likelihood.number_counts._source import NumberCounts
from firecrown.likelihood.number_counts._factories import (
    NumberCountsFactory,
)
from firecrown.likelihood.number_counts._systematics import (
    ConstantMagnificationBiasSystematic,
    MagnificationBiasSystematic,
    PhotoZShift,
    PTNonLinearBiasSystematic,
    PhotoZShiftandStretch,
)

# Re-export shared factories from base module
from firecrown.likelihood._base import (
    SpecZStretch,
    SpecZStretchFactory,
)

__all__ = [
    "NumberCounts",
    "NumberCountsFactory",
    "PhotoZShift",
    "ConstantMagnificationBiasSystematic",
    "PTNonLinearBiasSystematic",
    "MagnificationBiasSystematic",
    "PhotoZShiftandStretch",
    "SpecZStretch",
    "SpecZStretchFactory",
]
