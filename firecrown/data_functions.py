"""This module deals with two-point data functions.

It contains functions to manipulate two-point data objects.
"""

import hashlib
from typing import Callable, Sequence

import sacc

from firecrown.metadata_types import (
    TwoPointHarmonic,
    TwoPointReal,
)
from firecrown.metadata_functions import (
    extract_all_tracers_inferred_galaxy_zdists,
    extract_window_function,
    extract_all_harmonic_metadata_indices,
    extract_all_real_metadata_indices,
    make_two_point_xy,
)
from firecrown.data_types import TwoPointMeasurement


def extract_all_harmonic_data(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
    include_maybe_types=False,
) -> list[TwoPointMeasurement]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, include_maybe_types=include_maybe_types
        )
    }

    if sacc_data.covariance is None or sacc_data.covariance.dense is None:
        raise ValueError("The SACC object does not have a covariance matrix.")
    cov_hash = hashlib.sha256(sacc_data.covariance.dense).hexdigest()

    tpms: list[TwoPointMeasurement] = []
    for cell_index in extract_all_harmonic_metadata_indices(
        sacc_data, allowed_data_type
    ):
        t1, t2 = cell_index["tracer_names"]
        dt = cell_index["data_type"]

        ells, Cells, indices = sacc_data.get_ell_cl(
            data_type=dt, tracer1=t1, tracer2=t2, return_cov=False, return_ind=True
        )

        replacement_ells, weights = extract_window_function(sacc_data, indices)
        if replacement_ells is not None:
            ells = replacement_ells

        tpms.append(
            TwoPointMeasurement(
                data=Cells,
                indices=indices,
                covariance_name=cov_hash,
                metadata=TwoPointHarmonic(
                    XY=make_two_point_xy(
                        inferred_galaxy_zdists_dict, cell_index["tracer_names"], dt
                    ),
                    window=weights,
                    ells=ells,
                ),
            ),
        )

    return tpms


# Extracting the two-point function metadata and data from a sacc file


def extract_all_real_data(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
    include_maybe_types=False,
) -> list[TwoPointMeasurement]:
    """Extract the two-point function metadata and data from a sacc file."""
    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, include_maybe_types=include_maybe_types
        )
    }

    cov_hash = hashlib.sha256(sacc_data.covariance.dense).hexdigest()

    result: list[TwoPointMeasurement] = []
    for real_index in extract_all_real_metadata_indices(sacc_data, allowed_data_type):
        t1, t2 = real_index["tracer_names"]
        dt = real_index["data_type"]

        thetas, Xis, indices = sacc_data.get_theta_xi(
            data_type=dt, tracer1=t1, tracer2=t2, return_cov=False, return_ind=True
        )

        result.append(
            TwoPointMeasurement(
                data=Xis,
                indices=indices,
                covariance_name=cov_hash,
                metadata=TwoPointReal(
                    XY=make_two_point_xy(
                        inferred_galaxy_zdists_dict, real_index["tracer_names"], dt
                    ),
                    thetas=thetas,
                ),
            )
        )

    return result


def ensure_no_overlaps(
    measurement: TwoPointMeasurement,
    index_set: set[int],
    index_sets: list[set[int]],
    other_measurements: Sequence[TwoPointMeasurement],
) -> None:
    """Check if the indices of the measurement-space two-point functions overlap.

    Raises a ValueError if they do.

    :param measurement: The TwoPointHarmonic to check.
    :param index_set: The indices of the current TwoPointHarmonic.
    :param index_sets: The indices of the other TwoPointHarmonics.
    :param other_measurements: The other TwoPointHarmonics.
    """
    for i, one_set in enumerate(index_sets):
        if one_set & index_set:
            raise ValueError(
                f"The indices of the TwoPointHarmonic "
                f"{other_measurements[i]} and {measurement} overlap."
            )


def check_consistence(
    measurements: Sequence[TwoPointMeasurement],
    is_type_func: Callable[[TwoPointMeasurement], bool],
    type_name: str,
) -> None:
    """Check the indices of the two-point functions.

    Make sure the indices of the two-point functions are consistent.

    :param measurements: The measurements to check.
    :param is_type_func: A function to verify the type of the measurements.
    :param type_name: The type of the measurements.
    """
    seen_indices: set[int] = set()
    index_sets = []
    cov_name: None | str = None

    for measurement in measurements:
        if not is_type_func(measurement):
            raise ValueError(
                f"The metadata of the TwoPointMeasurement {measurement} is not "
                f"a measurement of {type_name}."
            )
        if cov_name is None:
            cov_name = measurement.covariance_name
        elif cov_name != measurement.covariance_name:
            raise ValueError(
                f"The {type_name} {measurement} has a different covariance "
                f"name {measurement.covariance_name} than the previous "
                f"{type_name} {cov_name}."
            )
        index_set = set(measurement.indices)
        index_sets.append(index_set)
        if len(index_set) != len(measurement.indices):
            raise ValueError(
                f"The indices of the {type_name} {measurement} are not unique."
            )

        if seen_indices & index_set:
            ensure_no_overlaps(measurement, index_set, index_sets, measurements)
        seen_indices.update(index_set)


def check_two_point_consistence_harmonic(
    two_point_harmonics: Sequence[TwoPointMeasurement],
) -> None:
    """Check the indices of the harmonic-space two-point functions."""
    check_consistence(
        two_point_harmonics, lambda m: m.is_harmonic(), "TwoPointHarmonic"
    )


def check_two_point_consistence_real(
    two_point_reals: Sequence[TwoPointMeasurement],
) -> None:
    """Check the indices of the real-space two-point functions."""
    check_consistence(two_point_reals, lambda m: m.is_real(), "TwoPointReal")
