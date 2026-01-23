"""This module deals with metadata types.

This module contains metadata types definitions.
"""

# Import all public types and classes from private submodules
from firecrown.metadata_types._compatibility import (
    measurement_is_compatible,
    measurements_types,
)
from firecrown.metadata_types._two_point_tracers import (
    TomographicBin,
    ProjectedField,
    CMBLensing,
    InferredGalaxyZDist,
)

# Backwards compatibility: expose the legacy name
from firecrown.metadata_types._measurements import (
    ALL_MEASUREMENTS,
    ALL_MEASUREMENT_TYPES,
    CMB,
    CMB_TYPES,
    GALAXY_LENS_TYPES,
    GALAXY_SOURCE_TYPES,
    LENS_REGEX,
    SOURCE_REGEX,
    Clusters,
    Galaxies,
    Measurement,
)
from firecrown.metadata_types._pair_selector import (
    AndBinPairSelector,
    AutoBinPairSelector,
    AutoMeasurementBinPairSelector,
    AutoNameBinPairSelector,
    AutoNameDiffBinPairSelector,
    BinPairSelector,
    CompositeSelector,
    CrossBinPairSelector,
    CrossMeasurementBinPairSelector,
    CrossNameBinPairSelector,
    CrossNameDiffBinPairSelector,
    LensBinPairSelector,
    MeasurementPair,
    NamedBinPairSelector,
    NotBinPairSelector,
    OrBinPairSelector,
    SourceBinPairSelector,
    SourceLensBinPairSelector,
    ThreeTwoBinPairSelector,
    TypeSourceBinPairSelector,
    TomographicBinPair,
    register_bin_pair_selector,
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
    "GALAXY_LENS_TYPES",
    "GALAXY_SOURCE_TYPES",
    "LENS_REGEX",
    "SOURCE_REGEX",
    # Utility types
    "TracerNames",
    "TRACER_NAMES_TOTAL",
    "TypeSource",
    # Tomographic bin / legacy name
    "TomographicBin",
    "InferredGalaxyZDist",
    "ProjectedField",
    "CMBLensing",
    # Two-point types
    "TwoPointXY",
    "TwoPointHarmonic",
    "TwoPointReal",
    "TwoPointCorrelationSpace",
    "TwoPointFilterMethod",
    # Compatibility function (public)
    "measurement_is_compatible",
    "measurements_types",
    # SACC conversion constant
    "MEASURED_TYPE_STRING_MAP",
    # Bin rule types and functions
    "AndBinPairSelector",
    "AutoBinPairSelector",
    "AutoMeasurementBinPairSelector",
    "AutoNameBinPairSelector",
    "AutoNameDiffBinPairSelector",
    "BinPairSelector",
    "CompositeSelector",
    "CrossBinPairSelector",
    "CrossMeasurementBinPairSelector",
    "CrossNameBinPairSelector",
    "CrossNameDiffBinPairSelector",
    "LensBinPairSelector",
    "NamedBinPairSelector",
    "NotBinPairSelector",
    "OrBinPairSelector",
    "SourceBinPairSelector",
    "SourceLensBinPairSelector",
    "ThreeTwoBinPairSelector",
    "TypeSourceBinPairSelector",
    "TomographicBinPair",
    "MeasurementPair",
    "register_bin_pair_selector",
]
