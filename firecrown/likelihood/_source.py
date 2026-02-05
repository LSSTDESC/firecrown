"""Abstract base classes for TwoPoint Statistics sources."""

from __future__ import annotations

# Import base classes from _base.py
from firecrown.likelihood._base import (
    SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z,
    SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z,
    SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_V,
    PhotoZShift,
    PhotoZShiftFactory,
    PhotoZShiftandStretch,
    PhotoZShiftandStretchFactory,
    SpecZStretch,
    SpecZStretchFactory,
    Source,
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxyPhotoZShift,
    SourceGalaxyPhotoZShiftandStretch,
    SourceGalaxySpecZStretch,
    SourceGalaxySelectField,
    SourceGalaxySystematic,
    SourceSystematic,
    Tracer,
    dndz_shift_and_stretch_active,
    dndz_shift_and_stretch_passive,
    dndz_stretch_fog_gaussian,
    dndz_stretch_fog_lorentzian,
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
    "SourceGalaxySpecZStretch",
    "SourceGalaxySelectField",
    "SourceGalaxy",
    "PhotoZShift",
    "PhotoZShiftFactory",
    "PhotoZShiftandStretch",
    "PhotoZShiftandStretchFactory",
    "SpecZStretch",
    "SpecZStretchFactory",
    "dndz_shift_and_stretch_active",
    "dndz_shift_and_stretch_passive",
    "dndz_stretch_fog_gaussian",
    "dndz_stretch_fog_lorentzian",
    "SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z",
    "SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z",
    "SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_V",
]
