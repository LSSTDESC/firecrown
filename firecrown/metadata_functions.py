"""This module deals with two-point functions metadata.

It contains all data classes and functions for store and extract two-point functions
metadata from a sacc file.
"""

from itertools import combinations_with_replacement, product
import hashlib
from typing import TypedDict, Sequence

import numpy as np
import numpy.typing as npt
import sacc
from sacc.data_types import required_tags

from firecrown.metadata_types import (
    TracerNames,
    Measurement,
    InferredGalaxyZDist,
    TwoPointXY,
    TwoPointHarmonic,
    TwoPointReal,
    TwoPointMeasurement,
    LENS_REGEX,
    SOURCE_REGEX,
    MEASURED_TYPE_STRING_MAP,
    measurement_is_compatible,
    GALAXY_LENS_TYPES,
    GALAXY_SOURCE_TYPES,
)


# TwoPointXiThetaIndex is a type used to create intermediate objects when
# reading SACC objects. They should not be seen directly by users of Firecrown.
TwoPointXiThetaIndex = TypedDict(
    "TwoPointXiThetaIndex",
    {
        "data_type": str,
        "tracer_names": TracerNames,
        "thetas": npt.NDArray[np.float64],
    },
)


# TwoPointCellsIndex is a type used to create intermediate objects when reading
# SACC objects. They should not be seen directly by users of Firecrown.
TwoPointCellsIndex = TypedDict(
    "TwoPointCellsIndex",
    {
        "data_type": str,
        "tracer_names": TracerNames,
        "ells": npt.NDArray[np.int64],
    },
)


def _extract_all_candidate_data_types(
    data_points: list[sacc.DataPoint],
    include_maybe_types: bool = False,
) -> dict[str, set[Measurement]]:
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
        if data_type not in MEASURED_TYPE_STRING_MAP:
            continue
        a, b = MEASURED_TYPE_STRING_MAP[data_type]

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
    sure_types: dict[str, set[Measurement]] = {}
    maybe_types: dict[str, set[Measurement]] = {}

    for data_type, tracer1, tracer2 in all_data_types:
        sure_types[tracer1] = set()
        sure_types[tracer2] = set()
        maybe_types[tracer1] = set()
        maybe_types[tracer2] = set()

    # Getting the sure and maybe types for each tracer.
    for data_type, tracer1, tracer2 in all_data_types:
        if data_type not in MEASURED_TYPE_STRING_MAP:
            continue
        a, b = MEASURED_TYPE_STRING_MAP[data_type]

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
    a: Measurement,
    b: Measurement,
    require_convetion: bool = False,
) -> tuple[bool, str, Measurement, str, Measurement]:
    """Use the naming convention to assign the right measurement to each tracer."""
    for n1, n2 in ((tracer1, tracer2), (tracer2, tracer1)):
        if LENS_REGEX.match(n1) and SOURCE_REGEX.match(n2):
            if a in GALAXY_SOURCE_TYPES and b in GALAXY_LENS_TYPES:
                return True, n1, b, n2, a
            if b in GALAXY_SOURCE_TYPES and a in GALAXY_LENS_TYPES:
                return True, n1, a, n2, b
            raise ValueError(
                "Invalid SACC file, tracer names do not respect "
                "the naming convetion."
            )
    if require_convetion:
        if LENS_REGEX.match(tracer1) and LENS_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b
        if SOURCE_REGEX.match(tracer1) and SOURCE_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b

        raise ValueError(
            f"Invalid tracer names ({tracer1}, {tracer2}) "
            f"do not respect the naming convetion."
        )

    return False, tracer1, a, tracer2, b


def extract_all_tracers(
    sacc_data: sacc.Sacc, include_maybe_types=False
) -> list[InferredGalaxyZDist]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    tracers: list[sacc.tracers.BaseTracer] = sacc_data.tracers.values()
    tracer_types = extract_all_tracers_types(
        sacc_data, include_maybe_types=include_maybe_types
    )
    for tracer0, tracer_types0 in tracer_types.items():
        if len(tracer_types0) == 0:
            raise ValueError(
                f"Tracer {tracer0} does not have data points associated with it. "
                f"Inconsistent SACC object."
            )

    return [
        InferredGalaxyZDist(
            bin_name=tracer.name,
            z=tracer.z,
            dndz=tracer.nz,
            measurements=tracer_types[tracer.name],
        )
        for tracer in tracers
    ]


def extract_all_tracers_types(
    sacc_data: sacc.Sacc,
    include_maybe_types: bool = False,
) -> dict[str, set[Measurement]]:
    """Extracts the two-point function metadata from a Sacc object.

    The Sacc object contains a set of tracers (one-dimensional bins) and data
    points (measurements of the correlation between two tracers).

    This function extracts the two-point function metadata from the Sacc object
    and returns it in a list.
    """
    data_points = sacc_data.get_data_points()

    return _extract_all_candidate_data_types(data_points, include_maybe_types)


def extract_all_data_types_reals(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
) -> list[TwoPointXiThetaIndex]:
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

    all_reals: list[TwoPointXiThetaIndex] = []
    for data_type in data_types_reals:
        for combo in sacc_data.get_tracer_combinations(data_type):
            if len(combo) != 2:
                raise ValueError(
                    f"Tracer combination {combo} does not have exactly two tracers."
                )

            all_reals.append(
                {
                    "data_type": data_type,
                    "tracer_names": TracerNames(*combo),
                    "thetas": np.array(
                        sacc_data.get_tag(tag_name, data_type=data_type, tracers=combo)
                    ),
                }
            )

    return all_reals


def extract_all_data_types_cells(
    sacc_data: sacc.Sacc, allowed_data_type: None | list[str] = None
) -> list[TwoPointCellsIndex]:
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

    all_cell: list[TwoPointCellsIndex] = []
    for data_type in data_types_cells:
        for combo in sacc_data.get_tracer_combinations(data_type):
            if len(combo) != 2:
                raise ValueError(
                    f"Tracer combination {combo} does not have exactly two tracers."
                )

            all_cell.append(
                {
                    "data_type": data_type,
                    "tracer_names": TracerNames(*combo),
                    "ells": np.array(
                        sacc_data.get_tag(tag_name, data_type=data_type, tracers=combo)
                    ).astype(np.int64),
                }
            )

    return all_cell


def extract_all_photoz_bin_combinations(
    sacc_data: sacc.Sacc,
    include_maybe_types: bool = False,
) -> list[TwoPointXY]:
    """Extracts the two-point function metadata from a sacc file."""
    inferred_galaxy_zdists = extract_all_tracers(
        sacc_data, include_maybe_types=include_maybe_types
    )
    bin_combinations = make_all_photoz_bin_combinations(inferred_galaxy_zdists)

    return bin_combinations


def extract_window_function(
    sacc_data: sacc.Sacc, indices: npt.NDArray[np.int64]
) -> tuple[None | npt.NDArray[np.float64], None | npt.NDArray[np.float64]]:
    """Extract ells and weights for a window function.

    :params sacc_data: the Sacc object from which we read.
    :params indices: the indices of the data points in the Sacc object which
        are computed by the window function.
    :returns: the ells and weights of the window function that match the
       given indices from a sacc object, or a tuple of (None, None)
       if the indices represent the measured Cells directly.
    """
    bandpower_window = sacc_data.get_bandpower_windows(indices)
    if bandpower_window is None:
        return None, None
    ells = bandpower_window.values
    weights = bandpower_window.weight / bandpower_window.weight.sum(axis=0)
    return ells, weights


def _build_two_point_xy(
    inferred_galaxy_zdists_dict, tracer_names, data_type
) -> TwoPointXY:
    """Build a TwoPointXY object from the inferred galaxy z distributions.

    The TwoPointXY object is built from the inferred galaxy z distributions, the data
    type, and the tracer names.
    """
    a, b = MEASURED_TYPE_STRING_MAP[data_type]

    igz1 = inferred_galaxy_zdists_dict[tracer_names[0]]
    igz2 = inferred_galaxy_zdists_dict[tracer_names[1]]

    ab = a in igz1.measurements and b in igz2.measurements
    ba = b in igz1.measurements and a in igz2.measurements
    if a != b and ab and ba:
        raise ValueError(
            f"Ambiguous measurements for tracers {tracer_names}. "
            f"Impossible to determine which measurement is from which tracer."
        )
    XY = TwoPointXY(
        x=igz1, y=igz2, x_measurement=a if ab else b, y_measurement=b if ab else a
    )

    return XY


def extract_all_data_cells(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
    include_maybe_types=False,
) -> list[TwoPointHarmonic]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers(
            sacc_data, include_maybe_types=include_maybe_types
        )
    }

    if sacc_data.covariance is None or sacc_data.covariance.dense is None:
        raise ValueError("The SACC object does not have a covariance matrix.")
    cov_hash = hashlib.sha256(sacc_data.covariance.dense).hexdigest()

    result: list[TwoPointHarmonic] = []
    for cell_index in extract_all_data_types_cells(sacc_data, allowed_data_type):
        tracer_names = cell_index["tracer_names"]
        ells = cell_index["ells"]
        data_type = cell_index["data_type"]

        XY = _build_two_point_xy(inferred_galaxy_zdists_dict, tracer_names, data_type)

        ells, Cells, indices = sacc_data.get_ell_cl(
            data_type=data_type,
            tracer1=tracer_names[0],
            tracer2=tracer_names[1],
            return_cov=False,
            return_ind=True,
        )

        replacement_ells, weights = extract_window_function(sacc_data, indices)
        if replacement_ells is not None:
            ells = replacement_ells

        result.append(
            TwoPointHarmonic(
                XY=XY,
                window=weights,
                ells=ells,
                Cell=TwoPointMeasurement(
                    data=Cells,
                    indices=indices,
                    covariance_name=cov_hash,
                ),
            )
        )

    return result


def extract_all_data_reals(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
    include_maybe_types=False,
) -> list[TwoPointReal]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers(
            sacc_data, include_maybe_types=include_maybe_types
        )
    }

    cov_hash = hashlib.sha256(sacc_data.covariance.dense).hexdigest()

    two_point_reals = []
    for real_index in extract_all_data_types_reals(sacc_data, allowed_data_type):
        tracer_names = real_index["tracer_names"]
        thetas = real_index["thetas"]
        data_type = real_index["data_type"]

        XY = _build_two_point_xy(inferred_galaxy_zdists_dict, tracer_names, data_type)

        thetas, Xis, indices = sacc_data.get_theta_xi(
            data_type=data_type,
            tracer1=tracer_names[0],
            tracer2=tracer_names[1],
            return_cov=False,
            return_ind=True,
        )

        Xi = TwoPointMeasurement(
            data=Xis,
            indices=indices,
            covariance_name=cov_hash,
        )

        two_point_reals.append(TwoPointReal(XY=XY, thetas=thetas, xis=Xi))

    return two_point_reals


def check_two_point_consistence_harmonic(
    two_point_harmonics: Sequence[TwoPointHarmonic],
) -> None:
    """Check the indices of the harmonic-space two-point functions.

    Make sure the indices of the harmonic-space two-point functions are consistent.
    """
    all_indices_set: set[int] = set()
    index_set_list = []
    cov_name: None | str = None

    for harmonic in two_point_harmonics:
        if harmonic.Cell is None:
            raise ValueError(
                f"The TwoPointHarmonic {harmonic} does not contain a data."
            )
        if cov_name is None:
            cov_name = harmonic.Cell.covariance_name
        elif cov_name != harmonic.Cell.covariance_name:
            raise ValueError(
                f"The TwoPointHarmonic {harmonic} has a different covariance "
                f"name {harmonic.Cell.covariance_name} than the previous "
                f"TwoPointHarmonic {cov_name}."
            )
        index_set = set(harmonic.Cell.indices)
        index_set_list.append(index_set)
        if len(index_set) != len(harmonic.Cell.indices):
            raise ValueError(
                f"The indices of the TwoPointHarmonic {harmonic} are not unique."
            )

        if all_indices_set & index_set:
            for i, index_set_a in enumerate(index_set_list):
                if index_set_a & index_set:
                    raise ValueError(
                        f"The indices of the TwoPointHarmonic "
                        f"{two_point_harmonics[i]} and {harmonic} overlap."
                    )
        all_indices_set.update(index_set)


def check_two_point_consistence_real(
    two_point_reals: Sequence[TwoPointReal],
) -> None:
    """Check the indices of the real-space two-point functions.

    Make sure the indices of the real-space two-point functions are consistent.
    """
    all_indices_set: set[int] = set()
    index_set_list = []
    cov_name: None | str = None

    for two_point_real in two_point_reals:
        if two_point_real.xis is None:
            raise ValueError(
                f"The TwoPointReal {two_point_real} does not contain a data."
            )
        if cov_name is None:
            cov_name = two_point_real.xis.covariance_name
        elif cov_name != two_point_real.xis.covariance_name:
            raise ValueError(
                f"The TwoPointReal {two_point_real} has a different covariance "
                f"name {two_point_real.xis.covariance_name} than the previous "
                f"TwoPointReal {cov_name}."
            )
        index_set = set(two_point_real.xis.indices)
        index_set_list.append(index_set)
        if len(index_set) != len(two_point_real.xis.indices):
            raise ValueError(
                f"The indices of the TwoPointReal {two_point_real} " f"are not unique."
            )

        if all_indices_set & index_set:
            for i, index_set_a in enumerate(index_set_list):
                if index_set_a & index_set:
                    raise ValueError(
                        f"The indices of the TwoPointReal {two_point_reals[i]} "
                        f"and {two_point_real} overlap."
                    )
        all_indices_set.update(index_set)


def make_all_photoz_bin_combinations(
    inferred_galaxy_zdists: list[InferredGalaxyZDist],
) -> list[TwoPointXY]:
    """Extract the two-point function metadata from a sacc file."""
    bin_combinations = [
        TwoPointXY(
            x=igz1, y=igz2, x_measurement=x_measurement, y_measurement=y_measurement
        )
        for igz1, igz2 in combinations_with_replacement(inferred_galaxy_zdists, 2)
        for x_measurement, y_measurement in product(
            igz1.measurements, igz2.measurements
        )
        if measurement_is_compatible(x_measurement, y_measurement)
    ]

    return bin_combinations


def measurements_from_index(
    index: TwoPointXiThetaIndex | TwoPointCellsIndex,
) -> tuple[str, Measurement, str, Measurement]:
    """Return the measurements from a TwoPointXiThetaIndex object."""
    a, b = MEASURED_TYPE_STRING_MAP[index["data_type"]]
    _, n1, a, n2, b = match_name_type(
        index["tracer_names"].name1,
        index["tracer_names"].name2,
        a,
        b,
        require_convetion=True,
    )
    return n1, a, n2, b
