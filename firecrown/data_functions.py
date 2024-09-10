"""This module deals with two-point data functions.

It contains functions to manipulate two-point data objects.
"""

import hashlib
from typing import Sequence

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


def check_two_point_consistence_harmonic(
    two_point_harmonics: Sequence[TwoPointMeasurement],
) -> None:
    """Check the indices of the harmonic-space two-point functions.

    Make sure the indices of the harmonic-space two-point functions are consistent.
    """
    all_indices_set: set[int] = set()
    index_set_list = []
    cov_name: None | str = None

    for harmonic in two_point_harmonics:
        if not harmonic.is_harmonic():
            raise ValueError(
                f"The metadata of the TwoPointMeasurement {harmonic} is not "
                f"a measurement of TwoPointHarmonic."
            )
        if cov_name is None:
            cov_name = harmonic.covariance_name
        elif cov_name != harmonic.covariance_name:
            raise ValueError(
                f"The TwoPointHarmonic {harmonic} has a different covariance "
                f"name {harmonic.covariance_name} than the previous "
                f"TwoPointHarmonic {cov_name}."
            )
        index_set = set(harmonic.indices)
        index_set_list.append(index_set)
        if len(index_set) != len(harmonic.indices):
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
    two_point_reals: Sequence[TwoPointMeasurement],
) -> None:
    """Check the indices of the real-space two-point functions.

    Make sure the indices of the real-space two-point functions are consistent.
    """
    all_indices_set: set[int] = set()
    index_set_list = []
    cov_name: None | str = None

    for two_point_real in two_point_reals:
        if not two_point_real.is_real():
            raise ValueError(
                f"The metadata of the TwoPointMeasurement {two_point_real} is not "
                f"a measurement of TwoPointReal."
            )
        if cov_name is None:
            cov_name = two_point_real.covariance_name
        elif cov_name != two_point_real.covariance_name:
            raise ValueError(
                f"The TwoPointReal {two_point_real} has a different covariance "
                f"name {two_point_real.covariance_name} than the previous "
                f"TwoPointReal {cov_name}."
            )
        index_set = set(two_point_real.indices)
        index_set_list.append(index_set)
        if len(index_set) != len(two_point_real.indices):
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
