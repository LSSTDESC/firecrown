"""This module deals with metadata types.

This module contains metadata types definitions.
"""

# Import all public types and classes from private submodules
from firecrown.metadata_types._compatibility import measurement_is_compatible
from firecrown.metadata_types._inferred_galaxy_zdist import InferredGalaxyZDist
from firecrown.metadata_types._measurements import (
    ALL_MEASUREMENTS,
    ALL_MEASUREMENT_TYPES,
    CMB,
    CMB_TYPES,
    EXACT_MATCH_MEASUREMENTS,
    GALAXY_LENS_TYPES,
    GALAXY_SOURCE_TYPES,
    HARMONIC_ONLY_MEASUREMENTS,
    INCOMPATIBLE_MEASUREMENTS,
    LENS_REGEX,
    REAL_ONLY_MEASUREMENTS,
    SOURCE_REGEX,
    Clusters,
    Galaxies,
    Measurement,
)
from firecrown.metadata_types._sacc_type_string import MEASURED_TYPE_STRING_MAP
from firecrown.metadata_types._two_point_types import (
    TwoPointCorrelationSpace,
    TwoPointFilterMethod,
    TwoPointHarmonic,
    TwoPointReal,
    TwoPointXY,
)
from firecrown.metadata_types._utils import (
    TRACER_NAMES_TOTAL,
    TracerNames,
    TypeSource,
)

# Define __all__ for explicit API contract
__all__ = [
    # Measurement enums
    "CMB",
    "Clusters",
    "Galaxies",
    "Measurement",
    # Measurement constants
    "ALL_MEASUREMENTS",
    "ALL_MEASUREMENT_TYPES",
    "CMB_TYPES",
    "EXACT_MATCH_MEASUREMENTS",
    "GALAXY_LENS_TYPES",
    "GALAXY_SOURCE_TYPES",
    "HARMONIC_ONLY_MEASUREMENTS",
    "INCOMPATIBLE_MEASUREMENTS",
    "LENS_REGEX",
    "REAL_ONLY_MEASUREMENTS",
    "SOURCE_REGEX",
    # Utility types
    "TracerNames",
    "TRACER_NAMES_TOTAL",
    "TypeSource",
    # Inferred galaxy zdist
    "InferredGalaxyZDist",
    # Two-point types
    "TwoPointXY",
    "TwoPointHarmonic",
    "TwoPointReal",
    "TwoPointCorrelationSpace",
    "TwoPointFilterMethod",
    # Compatibility function (public)
    "measurement_is_compatible",
    # SACC conversion constant
    "MEASURED_TYPE_STRING_MAP",
]
