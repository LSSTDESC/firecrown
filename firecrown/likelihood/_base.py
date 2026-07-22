"""Compatibilty shim for likelihood_base.

This module re-exports all symbols from firecrown.likelihood_base for backward
compatibility with code that imports from firecrown.likelihood._base directly.

.. deprecated::
    Import from firecrown.likelihood_base directly instead of
    firecrown.likelihood._base. This shim exists only for compatibility and will
    be removed in a future version.
"""

from firecrown.likelihood_base import (
    SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z,
    SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z,
    GuardedStatistic,
    Likelihood,
    NamedParameters,
    PhotoZShift,
    PhotoZShiftandStretch,
    PhotoZShiftandStretchFactory,
    PhotoZShiftFactory,
    Source,
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxyPhotoZShift,
    SourceGalaxyPhotoZShiftandStretch,
    SourceGalaxySelectField,
    SourceGalaxySystematic,
    SourceSystematic,
    Statistic,
    StatisticUnreadError,
    Tracer,
    TrivialStatistic,
    dndz_shift_and_stretch_active,
    dndz_shift_and_stretch_passive,
)

__all__ = [
    "GuardedStatistic",
    "Likelihood",
    "NamedParameters",
    "Source",
    "SourceGalaxy",
    "SourceGalaxyArgs",
    "SourceGalaxyPhotoZShift",
    "SourceGalaxyPhotoZShiftandStretch",
    "SourceGalaxySelectField",
    "SourceGalaxySystematic",
    "SourceSystematic",
    "Statistic",
    "StatisticUnreadError",
    "Tracer",
    "TrivialStatistic",
    "PhotoZShift",
    "PhotoZShiftFactory",
    "PhotoZShiftandStretch",
    "PhotoZShiftandStretchFactory",
    "SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z",
    "SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z",
    "dndz_shift_and_stretch_active",
    "dndz_shift_and_stretch_passive",
]
