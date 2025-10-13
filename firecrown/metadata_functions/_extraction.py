"""Functions for extracting two-point metadata from SACC objects."""

import numpy as np
import numpy.typing as npt
import sacc
from sacc.data_types import required_tags

import firecrown.metadata_types as mdt
from firecrown.metadata_functions._type_defs import (
    TwoPointRealIndex,
    TwoPointHarmonicIndex,
)
from firecrown.metadata_functions._matching import match_name_type, make_two_point_xy
from firecrown.metadata_functions._combination_utils import (
    make_all_photoz_bin_combinations,
)


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
