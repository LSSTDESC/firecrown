"""Abstract base classes for TwoPoint Statistics sources."""

from __future__ import annotations

# Import base classes from _base.py
from firecrown.likelihood._base import (
    SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z,
    SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z,
    PhotoZShift,
    PhotoZShiftFactory,
    PhotoZShiftandStretch,
    PhotoZShiftandStretchFactory,
    Source,
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxyPhotoZShift,
    SourceGalaxyPhotoZShiftandStretch,
    SourceGalaxySelectField,
    SourceGalaxySystematic,
    SourceSystematic,
    Tracer,
    dndz_shift_and_stretch_active,
    dndz_shift_and_stretch_passive,
)

# All classes have been moved to _base.py
# This module now just re-exports them for backward compatibility

__all__ = [
    "Source",
    "SourceSystematic",
    "Tracer",
    "SourceGalaxyArgs",
    "SourceGalaxySystematic",
    "SourceGalaxyPhotoZShift",
    "SourceGalaxyPhotoZShiftandStretch",
    "SourceGalaxySelectField",
    "SourceGalaxy",
    "PhotoZShift",
    "PhotoZShiftFactory",
    "PhotoZShiftandStretch",
    "PhotoZShiftandStretchFactory",
    "dndz_shift_and_stretch_active",
    "dndz_shift_and_stretch_passive",
    "SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z",
    "SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z",
]
