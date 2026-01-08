"""Data extraction functions for two-point data from SACC files."""

import sacc

from firecrown.data_types import TwoPointMeasurement
from firecrown.data_functions._utils import cov_hash
from firecrown.metadata_functions import (
    extract_all_harmonic_metadata_indices,
    extract_all_real_metadata_indices,
    extract_all_tracers_inferred_galaxy_zdists,
    make_two_point_xy,
    maybe_enforce_window,
)
from firecrown.metadata_types import TwoPointHarmonic, TwoPointReal


def extract_all_harmonic_data(
    sacc_data: sacc.Sacc,
    allow_mixed_types: bool = False,
    allowed_data_type: None | list[str] = None,
) -> list[TwoPointMeasurement]:
    """Extract the two-point function metadata and data from a sacc file."""
    if sacc_data.covariance is None or sacc_data.covariance.dense is None:
        raise ValueError("The SACC object does not have a dense covariance matrix.")

    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, allow_mixed_types
        )
    }

    result: list[TwoPointMeasurement] = []
    for cell_index in extract_all_harmonic_metadata_indices(
        sacc_data, allow_mixed_types, allowed_data_type
    ):
        t1, t2 = cell_index["tracer_names"]
        dt = cell_index["data_type"]

        ells, Cells, indices = sacc_data.get_ell_cl(
            data_type=dt, tracer1=t1, tracer2=t2, return_cov=False, return_ind=True
        )

        ells, weights, window_ells = maybe_enforce_window(ells, indices, sacc_data)

        result.append(
            TwoPointMeasurement(
                data=Cells,
                indices=indices,
                covariance_name=cov_hash(sacc_data),
                metadata=TwoPointHarmonic(
                    XY=make_two_point_xy(
                        inferred_galaxy_zdists_dict, cell_index["tracer_names"], dt
                    ),
                    window=weights,
                    window_ells=window_ells,
                    ells=ells,
                ),
            ),
        )

    return result


def extract_all_real_data(
    sacc_data: sacc.Sacc,
    allow_mixed_types: bool = False,
    allowed_data_type: None | list[str] = None,
) -> list[TwoPointMeasurement]:
    """Extract the two-point function metadata and data from a sacc file."""
    if sacc_data.covariance is None or sacc_data.covariance.dense is None:
        raise ValueError("The SACC object does not have a dense covariance matrix.")

    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(sacc_data)
    }

    result: list[TwoPointMeasurement] = []
    for real_index in extract_all_real_metadata_indices(
        sacc_data, allow_mixed_types, allowed_data_type
    ):
        t1, t2 = real_index["tracer_names"]
        dt = real_index["data_type"]

        thetas, Xis, indices = sacc_data.get_theta_xi(
            data_type=dt, tracer1=t1, tracer2=t2, return_cov=False, return_ind=True
        )

        result.append(
            TwoPointMeasurement(
                data=Xis,
                indices=indices,
                covariance_name=cov_hash(sacc_data),
                metadata=TwoPointReal(
                    XY=make_two_point_xy(
                        inferred_galaxy_zdists_dict, real_index["tracer_names"], dt
                    ),
                    thetas=thetas,
                ),
            )
        )

    return result
