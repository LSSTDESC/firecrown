"""Two-point metadata functions module.

This module contains functions used to manipulate two-point metadata, including
extracting metadata from a sacc file and creating new metadata objects.
"""

from firecrown.metadata_functions._type_defs import (
    TwoPointRealIndex,
    TwoPointHarmonicIndex,
)
from firecrown.metadata_functions._measurement_utils import (
    make_measurement,
    make_measurements,
    make_measurement_dict,
    make_measurements_dict,
    make_correlation_space,
)
from firecrown.metadata_functions._extraction import (
    extract_all_tracers_inferred_galaxy_zdists,
    extract_all_measured_types,
    extract_all_real_metadata_indices,
    extract_all_harmonic_metadata_indices,
    extract_all_harmonic_metadata,
    extract_all_real_metadata,
    extract_all_photoz_bin_combinations,
    extract_window_function,
    maybe_enforce_window,
)
from firecrown.metadata_functions._combination_utils import (
    make_all_bin_rule_combinations,
    make_all_photoz_bin_combinations,
    make_all_photoz_bin_combinations_with_cmb,
    make_cmb_galaxy_combinations_only,
)
from firecrown.metadata_functions._matching import (
    make_two_point_xy,
    measurements_from_index,
)

__all__ = [
    "TwoPointRealIndex",
    "TwoPointHarmonicIndex",
    "make_measurement",
    "make_measurements",
    "make_measurement_dict",
    "make_measurements_dict",
    "make_correlation_space",
    "extract_all_tracers_inferred_galaxy_zdists",
    "extract_all_measured_types",
    "extract_all_real_metadata_indices",
    "extract_all_harmonic_metadata_indices",
    "extract_all_harmonic_metadata",
    "extract_all_real_metadata",
    "extract_all_photoz_bin_combinations",
    "extract_window_function",
    "maybe_enforce_window",
    "make_all_bin_rule_combinations",
    "make_all_photoz_bin_combinations",
    "make_all_photoz_bin_combinations_with_cmb",
    "make_cmb_galaxy_combinations_only",
    "make_two_point_xy",
    "measurements_from_index",
]
