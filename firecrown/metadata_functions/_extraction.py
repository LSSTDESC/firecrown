"""Functions for extracting two-point metadata from SACC objects."""

import warnings
import numpy as np
import numpy.typing as npt
import sacc
from sacc.data_types import required_tags

import firecrown.metadata_types as mdt
from firecrown.metadata_functions._type_defs import (
    TwoPointRealIndex,
    TwoPointHarmonicIndex,
)
from firecrown.metadata_functions._matching import make_two_point_xy
from firecrown.metadata_functions._combination_utils import (
    make_all_photoz_bin_combinations,
)


def extract_all_tracers_inferred_galaxy_zdists(
    sacc_data: sacc.Sacc,
    allow_mixed_types: bool = False,
) -> list[mdt.InferredGalaxyZDist]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    tracers: list[sacc.tracers.BaseTracer] = sacc_data.tracers.values()
    tracer_types, _ = extract_all_measured_types(
        sacc_data, allow_mixed_types=allow_mixed_types
    )
    for tracer0, tracer_types0 in tracer_types.items():
        if len(tracer_types0) == 0:
            raise ValueError(
                f"Tracer {tracer0} does not have data points associated with it. "
                f"Inconsistent SACC object."
            )

    return [
        mdt.InferredGalaxyZDist(
            bin_name=tracer.name,
            z=tracer.z,
            dndz=tracer.nz,
            measurements=tracer_types[tracer.name],
        )
        for tracer in tracers
        if isinstance(tracer, sacc.tracers.NZTracer)
    ]


def _sacc_convention_warning(
    tracer1: str, tracer2: str, data_type: str, a: mdt.Measurement, b: mdt.Measurement
) -> str:
    """Generate a deprecation warning for non-standard SACC tracer assignments.

    This warning is triggered when the function auto-corrects a SACC file that violates
    the naming convention (where measurement type order in data type strings should
    match the tracer order). The auto-correction swaps tracer labels to conform to the
    convention, but this behavior is deprecated and will be removed in a future
    release.

    :param tracer1: Name of the first tracer in the original SACC data.
    :param tracer2: Name of the second tracer in the original SACC data.
    :param data_type: The SACC data type string (e.g., 'galaxy_shear_xi_plus').
    :param a: The first measurement type associated with tracer1.
    :param b: The second measurement type associated with tracer2.
    :return: A formatted warning message explaining the issue and next steps.
    """
    return f"""
SACC Convention Violation Detected (DEPRECATED AUTO-FIX)

Firecrown detected an inconsistency in how measurement types are assigned to tracers.
Specifically, assigning measurement type '{a}' to tracer '{tracer1}' and measurement
type '{b}' to tracer '{tracer2}' would create mixed-type measurements (multiple distinct
measurement types in the same tomographic bin).

The data type string '{data_type}' follows the SACC naming convention, where the order
of measurement types in the string must match the order of tracers. However, your SACC
file/object appears to violate this convention.

AUTO-CORRECTION PERFORMED
Because allow_mixed_types=False (the default), Firecrown attempted to correct this by
swapping the tracer assignment, assuming the tracers were simply misaligned. This auto-
correction is a convenience feature for legacy SACC files that don't follow the
convention.

⚠️  DEPRECATION NOTICE ⚠️

This automatic correction will be REMOVED in a future release. Going forward, files
that violate the SACC convention will be interpreted as genuinely mixed-type
measurements and will either raise an error (if allow_mixed_types=False) or be
processed as-is (if allow_mixed_types=True).

RECOMMENDED ACTION
To future-proof your code, fix your SACC file to follow the naming convention. See the
documentation for detailed instructions and a repairing instructions:
    https://firecrown.readthedocs.io/en/latest/sacc_usage.html
"""


def _extract_data_types_from_sacc(
    sacc_data: sacc.Sacc,
) -> set[tuple[str, str, str]]:
    """Extract all unique (data_type, tracer1, tracer2) tuples from SACC data points.

    :param sacc_data: The SACC object to extract from.
    :return: Set of (data_type, tracer1, tracer2) tuples.
    """
    data_points = sacc_data.get_data_points()
    return {
        (d.data_type, d.tracers[0], d.tracers[1])
        for d in data_points
        if d.data_type in mdt.MEASURED_TYPE_STRING_MAP
    }


def _initialize_tracer_types(
    all_data_types: set[tuple[str, str, str]],
) -> dict[str, set[mdt.Measurement]]:
    """Initialize tracer_types dictionary with empty sets for all tracers.

    :param all_data_types: Set of (data_type, tracer1, tracer2) tuples.
    :return: Dictionary mapping tracer names to empty measurement sets.
    """
    tracer_types: dict[str, set[mdt.Measurement]] = {}
    for _, tracer1, tracer2 in all_data_types:
        tracer_types.setdefault(tracer1, set())
        tracer_types.setdefault(tracer2, set())
    return tracer_types


def _process_single_type_measurements(
    all_data_types: set[tuple[str, str, str]],
    tracer_types: dict[str, set[mdt.Measurement]],
) -> None:
    """Process measurements where both tracers have the same measurement type.

    Modifies tracer_types in place.

    :param all_data_types: Set of (data_type, tracer1, tracer2) tuples.
    :param tracer_types: Dictionary to update with detected measurement types.
    """
    for data_type, tracer1, tracer2 in all_data_types:
        if data_type not in mdt.MEASURED_TYPE_STRING_MAP:
            continue
        a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]
        if a == b:
            tracer_types[tracer1].update({a})
            tracer_types[tracer2].update({b})


def _should_swap_tracers_for_convention(
    a: mdt.Measurement,
    b: mdt.Measurement,
    tracer_types: dict[str, set[mdt.Measurement]],
    tracer1: str,
    tracer2: str,
) -> bool:
    """Determine if swapping tracers would reduce type mixing.

    :param a: First measurement type.
    :param b: Second measurement type.
    :param tracer_types: Current detected types for all tracers.
    :param tracer1: Name of first tracer.
    :param tracer2: Name of second tracer.
    :return: True if swapping would reduce mixing, False otherwise.
    """
    # Count new types in original configuration
    n_original = len({a} | tracer_types[tracer1]) + len({b} | tracer_types[tracer2])
    # Count new types in swapped configuration
    n_swapped = len({a} | tracer_types[tracer2]) + len({b} | tracer_types[tracer1])
    return n_original > n_swapped


def _process_two_type_measurements(
    all_data_types: set[tuple[str, str, str]],
    tracer_types: dict[str, set[mdt.Measurement]],
    allow_mixed_types: bool,
) -> None:
    """Process measurements where tracers have different measurement types.

    Attempts auto-correction if allow_mixed_types=False and convention violations
    are detected. Modifies tracer_types in place.

    :param all_data_types: Set of (data_type, tracer1, tracer2) tuples.
    :param tracer_types: Dictionary to update with detected measurement types.
    :param allow_mixed_types: Whether to allow mixed-type measurements.
    """
    for data_type, tracer1, tracer2 in all_data_types:
        if data_type not in mdt.MEASURED_TYPE_STRING_MAP:
            continue
        a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]
        if a != b:
            # Skip if types are already correctly assigned
            if (a in tracer_types[tracer1]) and (b in tracer_types[tracer2]):
                continue

            # Attempt auto-correction for convention violations
            if not allow_mixed_types:
                if _should_swap_tracers_for_convention(
                    a, b, tracer_types, tracer1, tracer2
                ):
                    warnings.warn(
                        _sacc_convention_warning(tracer1, tracer2, data_type, a, b),
                        DeprecationWarning,
                    )
                    a, b = b, a

            tracer_types[tracer1].update({a})
            tracer_types[tracer2].update({b})


def _validate_tracer_types(
    tracer_types: dict[str, set[mdt.Measurement]],
    allow_mixed_types: bool,
) -> None:
    """Validate that mixed-type measurements comply with allow_mixed_types setting.

    :param tracer_types: Dictionary mapping tracer names to measurement types.
    :param allow_mixed_types: Whether to allow mixed-type measurements.
    :raises ValueError: If a tracer has multiple measurement types and
        allow_mixed_types=False.
    """
    for tracer, measurements in tracer_types.items():
        has_mixed, list_types = mdt.measurements_types(measurements)
        if has_mixed and not allow_mixed_types:
            raise ValueError(
                f"Tracer '{tracer}' has multiple measurement types: "
                f"[{', '.join(list_types)}]. This may indicate inconsistent labeling "
                f"in the SACC file/object. If this is intentional (mixed-type "
                f"measurements), set allow_mixed_types=True. Otherwise, please verify "
                f"that measurements were correctly associated with tracers. See the "
                f"SACC convention documentation for repair instructions: "
                f"https://firecrown.readthedocs.io/en/latest/sacc_usage.html",
            )


def _check_tracer_swap_needed(
    a: mdt.Measurement,
    b: mdt.Measurement,
    tracer_types: dict[str, set[mdt.Measurement]],
    combo: tuple[str, str],
    allow_mixed_types: bool,
) -> bool:
    """Check if tracers need to be swapped to follow SACC convention.

    This function determines if the assignment of measurement types to tracers
    violates the SACC naming convention and whether swapping would fix it.

    :param a: First measurement type from data type string.
    :param b: Second measurement type from data type string.
    :param tracer_types: Dictionary mapping tracer names to their measurement types.
    :param combo: Tuple of (tracer1_name, tracer2_name).
    :param allow_mixed_types: Whether to allow mixed-type measurements.
    :return: True if tracers should be swapped, False otherwise.
    """
    return (
        (not allow_mixed_types)
        and ((a not in tracer_types[combo[0]]) or (b not in tracer_types[combo[1]]))
        and ((a in tracer_types[combo[1]]) and (b in tracer_types[combo[0]]))
    )


def extract_all_measured_types(
    sacc_data: sacc.Sacc, allow_mixed_types: bool = False
) -> tuple[dict[str, set[mdt.Measurement]], list[str]]:
    """Extract all Measurement types associated with each tracer from a SACC object.

    This function analyzes the SACC data points to determine which Measurement types
    are associated with each tracer (tomographic bin). Following the SACC convention, a
    tracer should typically be associated with only one type of measurement (e.g.,
    galaxy shear, galaxy counts, or CMB convergence), as these represent distinct
    observational probes.

    SACC Convention & Auto-Correction
    ---------------------------------
    SACC follows a strict naming convention for measurement types: the order of
    measurement types in a data type string must match the order of the associated
    tracers. This ensures unambiguous interpretation of two-point measurements.

    If your SACC file violates this convention (causing a tracer to have multiple
    measurement types), this function will attempt to auto-correct it by swapping
    tracer labels when allow_mixed_types=False. This auto-correction is provided as a
    convenience for legacy SACC files but is **deprecated** and will be removed in a
    future release.

    Behavior Summary
    ----------------
    - **allow_mixed_types=False (default)**: Raises an error if a tracer has multiple
      measurement types. However, if the error is due to convention violations, the
      function attempts auto-correction first and warns about the deprecated behavior.
    - **allow_mixed_types=True**: Permits mixed-type measurements in the same
      tomographic bin without raising an error.

    :param sacc_data: The SACC object containing tracers and data points.
    :param allow_mixed_types: Controls handling of mixed-type measurements.
        If False (default), raises an error when a tracer has multiple measurement
        types (unless auto-corrected). If True, allows mixed-type measurements without
        error.
    :return: Dictionary mapping tracer names to sets of Measurement types associated
        with that tracer.
    :raises ValueError: If a tracer has multiple measurement types and
        allow_mixed_types=False (after attempting auto-correction).

    See Also
    --------
    For detailed information about the SACC convention and how to fix violations:
        https://firecrown.readthedocs.io/en/latest/sacc_usage.html
    """
    # Extract data type information from SACC
    all_data_types = _extract_data_types_from_sacc(sacc_data)

    # Initialize measurement type tracking
    tracer_types = _initialize_tracer_types(all_data_types)

    # Process single-type measurements (a == b)
    _process_single_type_measurements(all_data_types, tracer_types)

    # Process two-type measurements (a != b) with auto-correction if needed
    _process_two_type_measurements(all_data_types, tracer_types, allow_mixed_types)

    # Validate result
    _validate_tracer_types(tracer_types, allow_mixed_types)

    return tracer_types, sorted([data_type for data_type, _, _ in all_data_types])


def extract_all_real_metadata_indices(
    sacc_data: sacc.Sacc,
    allow_mixed_types: bool = False,
    allowed_data_type: None | list[str] = None,
) -> list[TwoPointRealIndex]:
    """Extract all two-point function metadata from a sacc file.

    Extracts the two-point function measurement metadata for all measurements
    made in real space  from a Sacc object.
    """
    tag_name = "theta"
    tracer_types, data_types = extract_all_measured_types(sacc_data, allow_mixed_types)

    data_types_reals = [
        data_type for data_type in data_types if tag_name in required_tags[data_type]
    ]
    if allowed_data_type is not None:
        data_types_reals = [
            data_type
            for data_type in data_types_reals
            if data_type in allowed_data_type
        ]

    all_real_indices: list[TwoPointRealIndex] = []
    for data_type in data_types_reals:
        a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]
        for combo in sacc_data.get_tracer_combinations(data_type):
            if len(combo) != 2:
                raise ValueError(
                    f"Tracer combination {combo} does not have exactly two tracers."
                )
            if _check_tracer_swap_needed(a, b, tracer_types, combo, allow_mixed_types):
                # Swap the order of the tracer types due to the convention
                a, b = b, a

            all_real_indices.append(
                {
                    "data_type": data_type,
                    "tracer_names": mdt.TracerNames(*combo),
                    "tracer_types": (a, b),
                }
            )

    return all_real_indices


def extract_all_harmonic_metadata_indices(
    sacc_data: sacc.Sacc,
    allow_mixed_types: bool = False,
    allowed_data_type: None | list[str] = None,
) -> list[TwoPointHarmonicIndex]:
    """Extracts the two-point function metadata from a sacc file."""
    tag_name = "ell"
    tracer_types, data_types = extract_all_measured_types(sacc_data, allow_mixed_types)

    data_types_cells = [
        data_type for data_type in data_types if tag_name in required_tags[data_type]
    ]

    if allowed_data_type is not None:
        data_types_cells = [
            data_type
            for data_type in data_types_cells
            if data_type in allowed_data_type
        ]

    all_harmonic_indices: list[TwoPointHarmonicIndex] = []
    for data_type in data_types_cells:
        a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]
        for combo in sacc_data.get_tracer_combinations(data_type):
            if len(combo) != 2:
                raise ValueError(
                    f"Tracer combination {combo} does not have exactly two tracers."
                )
            if _check_tracer_swap_needed(a, b, tracer_types, combo, allow_mixed_types):
                # Swap the order of the tracer types due to the convention
                a, b = b, a

            all_harmonic_indices.append(
                {
                    "data_type": data_type,
                    "tracer_names": mdt.TracerNames(*combo),
                    "tracer_types": (a, b),
                }
            )

    return all_harmonic_indices


def extract_all_harmonic_metadata(
    sacc_data: sacc.Sacc,
    allow_mixed_types: bool = False,
    allowed_data_type: None | list[str] = None,
) -> list[mdt.TwoPointHarmonic]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, allow_mixed_types
        )
    }

    result: list[mdt.TwoPointHarmonic] = []
    for cell_index in extract_all_harmonic_metadata_indices(
        sacc_data, allow_mixed_types, allowed_data_type
    ):
        tracer_names = cell_index["tracer_names"]
        dt = cell_index["data_type"]

        XY = make_two_point_xy(inferred_galaxy_zdists_dict, tracer_names, dt)

        ells, _, indices = sacc_data.get_ell_cl(
            data_type=dt,
            tracer1=tracer_names[0],
            tracer2=tracer_names[1],
            return_cov=False,
            return_ind=True,
        )
        ells, weights, window_ells = maybe_enforce_window(ells, indices, sacc_data)

        result.append(
            mdt.TwoPointHarmonic(
                XY=XY,
                window=weights,
                window_ells=window_ells,
                ells=ells,
            )
        )

    return result


def extract_all_real_metadata(
    sacc_data: sacc.Sacc,
    allow_mixed_types: bool = False,
    allowed_data_type: None | list[str] = None,
) -> list[mdt.TwoPointReal]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, allow_mixed_types
        )
    }

    tprs: list[mdt.TwoPointReal] = []
    for real_index in extract_all_real_metadata_indices(
        sacc_data, allow_mixed_types, allowed_data_type
    ):
        tracer_names = real_index["tracer_names"]
        dt = real_index["data_type"]

        XY = make_two_point_xy(inferred_galaxy_zdists_dict, tracer_names, dt)

        t1, t2 = tracer_names
        thetas, _, _ = sacc_data.get_theta_xi(
            data_type=dt, tracer1=t1, tracer2=t2, return_cov=False, return_ind=True
        )

        tprs.append(mdt.TwoPointReal(XY=XY, thetas=thetas))

    return tprs


def extract_all_photoz_bin_combinations(
    sacc_data: sacc.Sacc, allow_mixed_types: bool = False
) -> list[mdt.TwoPointXY]:
    """Extracts the two-point function metadata from a sacc file."""
    inferred_galaxy_zdists = extract_all_tracers_inferred_galaxy_zdists(
        sacc_data, allow_mixed_types
    )
    bin_combinations = make_all_photoz_bin_combinations(inferred_galaxy_zdists)

    return bin_combinations


def extract_window_function(
    sacc_data: sacc.Sacc, indices: npt.NDArray[np.int64]
) -> tuple[None | npt.NDArray[np.int64], None | npt.NDArray[np.float64]]:
    """Extract ells and weights for a window function.

    :params sacc_data: the Sacc object from which we read.
    :params indices: the indices of the data points in the Sacc object which
        are computed by the window function.
    :return: the ells and weights of the window function that match the
       given indices from a sacc object, or a tuple of (None, None)
       if the indices represent the measured Cells directly.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No bandpower windows associated with these data",
        )
        bandpower_window = sacc_data.get_bandpower_windows(indices)
    if bandpower_window is None:
        return None, None
    ells = bandpower_window.values
    weights = bandpower_window.weight / bandpower_window.weight.sum(axis=0)
    return ells, weights


def maybe_enforce_window(
    ells: npt.NDArray, indices: npt.NDArray[np.int64], sacc_data: sacc.Sacc
) -> tuple[npt.NDArray[np.int64], None | npt.NDArray[np.float64], None | npt.NDArray]:
    """Possibly enforce a window function on the given ells.

    :param ells: The original ell values.
    :param indices: The indices of the data points in the SACC object.
    :param sacc_data: The SACC object containing the data.
    :return: A tuple containing the possibly replaced ells and the window weights.
    """
    replacement_ells, weights = extract_window_function(sacc_data, indices)
    if replacement_ells is not None:
        window_ells = ells
        ells = replacement_ells
    else:
        window_ells = None

    return ells, weights, window_ells
