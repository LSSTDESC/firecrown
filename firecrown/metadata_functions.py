"""This module deals with two-point metadata functions.

It contains functions used to manipulate two-point metadata, including extracting
metadata from a sacc file and creating new metadata objects.
"""

from itertools import combinations_with_replacement, product
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import sacc
from sacc.data_types import required_tags

import firecrown.metadata_types as mdt

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
    "make_all_photoz_bin_combinations",
    "make_all_photoz_bin_combinations_with_cmb",
    "make_cmb_galaxy_combinations_only",
    "match_name_type",
    "make_two_point_xy",
    "measurements_from_index",
]


# TwoPointRealIndex is a type used to create intermediate objects when reading SACC
# objects. They should not be seen directly by users of Firecrown.
class TwoPointRealIndex(TypedDict):
    """Intermediate object for reading SACC real-space two-point data.

    Internal use only - not intended for direct user interaction.
    """

    data_type: str
    tracer_names: mdt.TracerNames


# TwoPointHarmonicIndex is a type used to create intermediate objects when reading SACC
# objects. They should not be seen directly by users of Firecrown.
class TwoPointHarmonicIndex(TypedDict):
    """Intermediate object for reading SACC harmonic-space two-point data.

    Internal use only - not intended for direct user interaction.
    """

    data_type: str
    tracer_names: mdt.TracerNames


def make_measurement(value: mdt.Measurement | dict[str, Any]) -> mdt.Measurement:
    """Create a Measurement object from a dictionary."""
    if isinstance(value, mdt.ALL_MEASUREMENT_TYPES):
        return value

    if not isinstance(value, dict):
        raise ValueError(f"Invalid Measurement: {value} is not a dictionary")

    if "subject" not in value:
        raise ValueError("Invalid Measurement: dictionary does not contain 'subject'")

    subject = value["subject"]

    match subject:
        case "Galaxies":
            return mdt.Galaxies[value["property"]]
        case "CMB":
            return mdt.CMB[value["property"]]
        case "Clusters":
            return mdt.Clusters[value["property"]]
        case _:
            raise ValueError(
                f"Invalid Measurement: subject: '{subject}' is not recognized"
            )


def make_measurements(
    value: set[mdt.Measurement] | list[dict[str, Any]],
) -> set[mdt.Measurement]:
    """Create a Measurement object from a dictionary."""
    if isinstance(value, set) and all(
        isinstance(v, mdt.ALL_MEASUREMENT_TYPES) for v in value
    ):
        return value

    measurements: set[mdt.Measurement] = set()
    for measurement_dict in value:
        measurements.update([make_measurement(measurement_dict)])
    return measurements


def make_measurement_dict(value: mdt.Measurement) -> dict[str, str]:
    """Create a dictionary from a Measurement object.

    :param value: the measurement to turn into a dictionary
    """
    return {"subject": type(value).__name__, "property": value.name}


def make_measurements_dict(value: set[mdt.Measurement]) -> list[dict[str, str]]:
    """Create a dictionary from a Measurement object.

    :param value: the measurement to turn into a dictionary
    """
    return [make_measurement_dict(measurement) for measurement in value]


def make_correlation_space(value: mdt.TwoPointCorrelationSpace | str):
    """Create a CorrelationSpace object from a string."""
    if not isinstance(value, mdt.TwoPointCorrelationSpace) and isinstance(value, str):
        try:
            return mdt.TwoPointCorrelationSpace(
                value.lower()
            )  # Convert from string to Enum
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for TwoPointCorrelationSpace: {value}"
            ) from exc
    return value


def _extract_all_candidate_measurement_types(
    data_points: list[sacc.DataPoint],
    include_maybe_types: bool = False,
) -> dict[str, set[mdt.Measurement]]:
    """Extract all candidate Measurement from the data points.

    The candidate Measurement are the ones that appear in the data points.
    """
    all_data_types: set[tuple[str, str, str]] = {
        (d.data_type, d.tracers[0], d.tracers[1]) for d in data_points
    }
    sure_types, maybe_types = _extract_sure_and_maybe_types(all_data_types)

    # Remove the sure types from the maybe types.
    for tracer0, sure_types0 in sure_types.items():
        maybe_types[tracer0] -= sure_types0

    # Filter maybe types.
    for data_type, tracer1, tracer2 in all_data_types:
        if data_type not in mdt.MEASURED_TYPE_STRING_MAP:
            continue
        a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]

        if a == b:
            continue

        if a in sure_types[tracer1] and b in sure_types[tracer2]:
            maybe_types[tracer1].discard(b)
            maybe_types[tracer2].discard(a)
        elif a in sure_types[tracer2] and b in sure_types[tracer1]:
            maybe_types[tracer1].discard(a)
            maybe_types[tracer2].discard(b)

    if include_maybe_types:
        return {
            tracer0: sure_types0 | maybe_types[tracer0]
            for tracer0, sure_types0 in sure_types.items()
        }
    return sure_types


def _extract_sure_and_maybe_types(all_data_types):
    sure_types: dict[str, set[mdt.Measurement]] = {}
    maybe_types: dict[str, set[mdt.Measurement]] = {}

    for data_type, tracer1, tracer2 in all_data_types:
        sure_types.setdefault(tracer1, set())
        sure_types.setdefault(tracer2, set())
        maybe_types.setdefault(tracer1, set())
        maybe_types.setdefault(tracer2, set())

    # Getting the sure and maybe types for each tracer.
    for data_type, tracer1, tracer2 in all_data_types:
        if data_type not in mdt.MEASURED_TYPE_STRING_MAP:
            continue
        a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]

        if a == b:
            sure_types[tracer1].update({a})
            sure_types[tracer2].update({a})
        else:
            name_match, n1, a, n2, b = match_name_type(tracer1, tracer2, a, b)
            if name_match:
                sure_types[n1].update({a})
                sure_types[n2].update({b})
            if not name_match:
                maybe_types[tracer1].update({a, b})
                maybe_types[tracer2].update({a, b})
    return sure_types, maybe_types


def match_name_type(
    tracer1: str,
    tracer2: str,
    a: mdt.Measurement,
    b: mdt.Measurement,
    require_convetion: bool = False,
) -> tuple[bool, str, mdt.Measurement, str, mdt.Measurement]:
    """Use the naming convention to assign the right measurement to each tracer."""
    for n1, n2 in ((tracer1, tracer2), (tracer2, tracer1)):
        if mdt.LENS_REGEX.match(n1) and mdt.SOURCE_REGEX.match(n2):
            if a in mdt.GALAXY_SOURCE_TYPES and b in mdt.GALAXY_LENS_TYPES:
                return True, n1, b, n2, a
            if b in mdt.GALAXY_SOURCE_TYPES and a in mdt.GALAXY_LENS_TYPES:
                return True, n1, a, n2, b
            raise ValueError(
                "Invalid SACC file, tracer names do not respect "
                "the naming convetion."
            )
    if require_convetion:
        if mdt.LENS_REGEX.match(tracer1) and mdt.LENS_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b
        if mdt.SOURCE_REGEX.match(tracer1) and mdt.SOURCE_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b

        raise ValueError(
            f"Invalid tracer names ({tracer1}, {tracer2}) "
            f"do not respect the naming convetion."
        )

    return False, tracer1, a, tracer2, b


def extract_all_tracers_inferred_galaxy_zdists(
    sacc_data: sacc.Sacc, include_maybe_types=False
) -> list[mdt.InferredGalaxyZDist]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    tracers: list[sacc.tracers.BaseTracer] = sacc_data.tracers.values()
    tracer_types = extract_all_measured_types(
        sacc_data, include_maybe_types=include_maybe_types
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
    ]


def extract_all_measured_types(
    sacc_data: sacc.Sacc,
    include_maybe_types: bool = False,
) -> dict[str, set[mdt.Measurement]]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    data_points = sacc_data.get_data_points()

    return _extract_all_candidate_measurement_types(data_points, include_maybe_types)


def extract_all_real_metadata_indices(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
) -> list[TwoPointRealIndex]:
    """Extract all two-point function metadata from a sacc file.

    Extracts the two-point function measurement metadata for all measurements
    made in real space  from a Sacc object.
    """
    tag_name = "theta"

    data_types = sacc_data.get_data_types()

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
        for combo in sacc_data.get_tracer_combinations(data_type):
            if len(combo) != 2:
                raise ValueError(
                    f"Tracer combination {combo} does not have exactly two tracers."
                )

            all_real_indices.append(
                {
                    "data_type": data_type,
                    "tracer_names": mdt.TracerNames(*combo),
                }
            )

    return all_real_indices


def extract_all_harmonic_metadata_indices(
    sacc_data: sacc.Sacc, allowed_data_type: None | list[str] = None
) -> list[TwoPointHarmonicIndex]:
    """Extracts the two-point function metadata from a sacc file."""
    tag_name = "ell"

    data_types = sacc_data.get_data_types()

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
        for combo in sacc_data.get_tracer_combinations(data_type):
            if len(combo) != 2:
                raise ValueError(
                    f"Tracer combination {combo} does not have exactly two tracers."
                )

            all_harmonic_indices.append(
                {
                    "data_type": data_type,
                    "tracer_names": mdt.TracerNames(*combo),
                }
            )

    return all_harmonic_indices


def extract_all_harmonic_metadata(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
    include_maybe_types=False,
) -> list[mdt.TwoPointHarmonic]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, include_maybe_types=include_maybe_types
        )
    }

    result: list[mdt.TwoPointHarmonic] = []
    for cell_index in extract_all_harmonic_metadata_indices(
        sacc_data, allowed_data_type
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


# Extracting all real metadata from a SACC object.


def extract_all_real_metadata(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
    include_maybe_types=False,
) -> list[mdt.TwoPointReal]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, include_maybe_types=include_maybe_types
        )
    }

    tprs: list[mdt.TwoPointReal] = []
    for real_index in extract_all_real_metadata_indices(sacc_data, allowed_data_type):
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
    sacc_data: sacc.Sacc,
    include_maybe_types: bool = False,
) -> list[mdt.TwoPointXY]:
    """Extracts the two-point function metadata from a sacc file."""
    inferred_galaxy_zdists = extract_all_tracers_inferred_galaxy_zdists(
        sacc_data, include_maybe_types=include_maybe_types
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


def make_all_photoz_bin_combinations(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
) -> list[mdt.TwoPointXY]:
    """Extract the two-point function metadata from a sacc file."""
    bin_combinations = [
        mdt.TwoPointXY(
            x=igz1, y=igz2, x_measurement=x_measurement, y_measurement=y_measurement
        )
        for igz1, igz2 in combinations_with_replacement(inferred_galaxy_zdists, 2)
        for x_measurement, y_measurement in product(
            igz1.measurements, igz2.measurements
        )
        if mdt.measurement_is_compatible(x_measurement, y_measurement)
    ]

    return bin_combinations


def make_all_photoz_bin_combinations_with_cmb(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
    cmb_tracer_name: str = "cmb_convergence",
    include_cmb_auto: bool = False,
) -> list[mdt.TwoPointXY]:
    """Create all galaxy combinations plus mdt.CMB-galaxy cross-correlations.

    :param inferred_galaxy_zdists: List of galaxy redshift bins
    :param cmb_tracer_name: Name of the mdt.CMB tracer
    :param include_cmb_auto: Whether to include mdt.CMB auto-correlation
        (default: False)
    :return: List of all XY combinations including mdt.CMB-galaxy crosses
    """
    # Get all galaxy-galaxy combinations first
    galaxy_combinations = make_all_photoz_bin_combinations(inferred_galaxy_zdists)

    # Create a mock mdt.CMB "bin" for cross-correlations
    cmb_bin = mdt.InferredGalaxyZDist(
        bin_name=cmb_tracer_name,
        z=np.array([1100.0]),  # mdt.CMB redshift
        dndz=np.array([1.0]),  # Unity normalization
        measurements={mdt.CMB.CONVERGENCE},
        type_source=mdt.TypeSource.DEFAULT,
    )

    # Create mdt.CMB-galaxy cross-correlations only
    cmb_galaxy_combinations = []

    for galaxy_bin in inferred_galaxy_zdists:
        for galaxy_measurement in galaxy_bin.measurements:
            # Only create cross-correlations that are physically meaningful
            if mdt.measurement_is_compatible(mdt.CMB.CONVERGENCE, galaxy_measurement):
                # mdt.CMB-galaxy cross-correlation
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(
                        x=cmb_bin,
                        y=galaxy_bin,
                        x_measurement=mdt.CMB.CONVERGENCE,
                        y_measurement=galaxy_measurement,
                    )
                )

                # Galaxy-mdt.CMB cross-correlation (symmetric)
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(
                        x=galaxy_bin,
                        y=cmb_bin,
                        x_measurement=galaxy_measurement,
                        y_measurement=mdt.CMB.CONVERGENCE,
                    )
                )

    # Optionally include mdt.CMB auto-correlation
    if include_cmb_auto:
        cmb_galaxy_combinations.append(
            mdt.TwoPointXY(
                x=cmb_bin,
                y=cmb_bin,
                x_measurement=mdt.CMB.CONVERGENCE,
                y_measurement=mdt.CMB.CONVERGENCE,
            )
        )

    return galaxy_combinations + cmb_galaxy_combinations


def make_cmb_galaxy_combinations_only(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
    cmb_tracer_name: str = "cmb_convergence",
) -> list[mdt.TwoPointXY]:
    """Create only mdt.CMB-galaxy cross-correlations.

    :param inferred_galaxy_zdists: List of galaxy redshift bins
    :param cmb_tracer_name: Name of the mdt.CMB tracer
    :return: List of mdt.CMB-galaxy cross-correlation XY combinations only
    """
    # Create a mock mdt.CMB "bin"
    cmb_bin = mdt.InferredGalaxyZDist(
        bin_name=cmb_tracer_name,
        z=np.array([1100.0]),
        dndz=np.array([1.0]),
        measurements={mdt.CMB.CONVERGENCE},
        type_source=mdt.TypeSource.DEFAULT,
    )

    cmb_galaxy_combinations = []

    for galaxy_bin in inferred_galaxy_zdists:
        for galaxy_measurement in galaxy_bin.measurements:
            if mdt.measurement_is_compatible(mdt.CMB.CONVERGENCE, galaxy_measurement):
                # mdt.CMB-galaxy cross-correlation
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(
                        x=cmb_bin,
                        y=galaxy_bin,
                        x_measurement=mdt.CMB.CONVERGENCE,
                        y_measurement=galaxy_measurement,
                    )
                )

                # Galaxy-mdt.CMB cross-correlation (symmetric)
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(
                        x=galaxy_bin,
                        y=cmb_bin,
                        x_measurement=galaxy_measurement,
                        y_measurement=mdt.CMB.CONVERGENCE,
                    )
                )

    return cmb_galaxy_combinations


def make_two_point_xy(
    inferred_galaxy_zdists_dict: dict[str, mdt.InferredGalaxyZDist],
    tracer_names: mdt.TracerNames,
    data_type: str,
) -> mdt.TwoPointXY:
    """Build a mdt.TwoPointXY object from the inferred galaxy z distributions.

    The mdt.TwoPointXY object is built from the inferred galaxy z distributions,
    the data type, and the tracer names.

    :param inferred_galaxy_zdists_dict: a dictionary of inferred galaxy z distributions.
    :param tracer_names: a tuple of tracer names.
    :param data_type: the data type.

    :return: a mdt.TwoPointXY object.
    """
    a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]

    igz1 = inferred_galaxy_zdists_dict[tracer_names[0]]
    igz2 = inferred_galaxy_zdists_dict[tracer_names[1]]

    ab = a in igz1.measurements and b in igz2.measurements
    ba = b in igz1.measurements and a in igz2.measurements
    if a != b and ab and ba:
        raise ValueError(
            f"Ambiguous measurements for tracers {tracer_names}. "
            f"Impossible to determine which measurement is from which tracer."
        )
    XY = mdt.TwoPointXY(
        x=igz1, y=igz2, x_measurement=a if ab else b, y_measurement=b if ab else a
    )

    return XY


def measurements_from_index(
    index: TwoPointRealIndex | TwoPointHarmonicIndex,
) -> tuple[str, mdt.Measurement, str, mdt.Measurement]:
    """Return the measurements from a TwoPointXiThetaIndex object."""
    a, b = mdt.MEASURED_TYPE_STRING_MAP[index["data_type"]]
    _, n1, a, n2, b = match_name_type(
        index["tracer_names"].name1,
        index["tracer_names"].name2,
        a,
        b,
        require_convetion=True,
    )
    return n1, a, n2, b
