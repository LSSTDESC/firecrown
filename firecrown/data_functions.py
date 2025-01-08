"""This module deals with two-point data functions.

It contains functions to manipulate two-point data objects.
"""

import hashlib
from typing import Sequence, Literal

import sacc
from pydantic import BaseModel, Field, model_validator, ConfigDict, PrivateAttr
import numpy as np
import numpy.typing as npt

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
        index_set: set[int] = set(harmonic.indices)
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
        index_set: set[int] = set(two_point_real.indices)
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


class TwoPointBinFilter(BaseModel):
    """Class defining a filter for a bin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bin_name: str = Field(
        description="The name of the bin to filter.",
    )
    bin_filter: tuple[float, float] = Field(
        description="The range of the bin to filter.",
    )

    @model_validator(mode="after")
    def check_bin_filter(self) -> "TwoPointBinFilter":
        """Check the bin filter."""
        if self.bin_filter[0] >= self.bin_filter[1]:
            raise ValueError("The bin filter should be a valid range.")
        return self


class TwoPointBinFilterCollection(BaseModel):
    """Class defining a collection of bin filters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bin_filters: list[TwoPointBinFilter] = Field(
        description="The list of bin filters.",
    )
    require_filter_for_all: bool = Field(
        default=False,
        description="If True, all bins should have a filter.",
    )
    filter_merge_mode: Literal["union", "intersection"] = Field(
        description="The mode to merge the filters.",
    )
    allow_empty: bool = Field(
        default=False,
        description=(
            "When true, objects with no elements remaining after applying "
            "the filter will be ignored rather than treated as an error."
        ),
    )

    _bin_filter_dict: dict[str, tuple[float, float]] = PrivateAttr()

    @model_validator(mode="after")
    def check_bin_filters(self) -> "TwoPointBinFilterCollection":
        """Check the bin filters."""
        bin_names = set()
        for bin_filter in self.bin_filters:
            if bin_filter.bin_name in bin_names:
                raise ValueError(
                    f"The bin name {bin_filter.bin_name} is repeated "
                    f"in the bin filters."
                )
            bin_names.add(bin_filter.bin_name)

        self._bin_filter_dict = {
            bin_filter.bin_name: bin_filter.bin_filter
            for bin_filter in self.bin_filters
        }
        return self

    @property
    def bin_filter_dict(self) -> dict[str, tuple[float, float]]:
        """Return the bin filter dictionary."""
        return self._bin_filter_dict

    def filter_match(self, tpm: TwoPointMeasurement) -> bool:
        """Check if the TwoPointMeasurement matches the filter."""
        if tpm.metadata.XY.x.bin_name not in self._bin_filter_dict:
            return False
        if tpm.metadata.XY.y.bin_name not in self._bin_filter_dict:
            return False
        return True

    def run_merge_filter(
        self,
        filter_x: tuple[float, float],
        filter_y: tuple[float, float],
        vals: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    ) -> npt.NDArray[np.bool_]:
        """Run the filter merge."""
        match self.filter_merge_mode:
            case "union":
                match_elements = (vals >= filter_x[0]) & (vals <= filter_x[1]) | (
                    vals >= filter_y[0]
                ) & (vals <= filter_y[1])

            case "intersection":
                match_elements = (
                    (vals >= filter_x[0])
                    & (vals <= filter_x[1])
                    & (vals >= filter_y[0])
                    & (vals <= filter_y[1])
                )
        return match_elements

    def apply_filter_single(
        self, tpm: TwoPointMeasurement
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Apply the filter to a single TwoPointMeasurement."""
        assert self.filter_match(tpm)
        filter_x = self._bin_filter_dict[tpm.metadata.XY.x.bin_name]
        filter_y = self._bin_filter_dict[tpm.metadata.XY.y.bin_name]
        if tpm.is_real():
            assert isinstance(tpm.metadata, TwoPointReal)
            match_elements = self.run_merge_filter(
                filter_x, filter_y, tpm.metadata.thetas
            )
            return match_elements, match_elements

        assert isinstance(tpm.metadata, TwoPointHarmonic)
        match_elements = self.run_merge_filter(filter_x, filter_y, tpm.metadata.ells)
        match_obs = match_elements
        if tpm.metadata.window is not None:
            # The window function is represented by a matrix where each column
            # corresponds to the weights for the ell values of each observation. We
            # need to ensure that the window function is filtered correctly. To do this,
            # we will check each column of the matrix and verify that all non-zero
            # elements are within the filtered set. If any non-zero element falls
            # outside the filtered set, the match_elements will be set to False for that
            # observation.
            non_zero_window = tpm.metadata.window > 0
            match_obs = (
                np.all(
                    (non_zero_window & match_elements[:, None]) == non_zero_window,
                    axis=0,
                )
                .ravel()
                .astype(np.bool_)
            )

        return match_elements, match_obs

    def __call__(
        self, tpms: Sequence[TwoPointMeasurement]
    ) -> list[TwoPointMeasurement]:
        """Filter the two-point measurements."""
        result = []

        for tpm in tpms:
            if not self.filter_match(tpm):
                if not self.require_filter_for_all:
                    continue
                raise ValueError(
                    f"The bin name {tpm.metadata.XY.get_tracer_names()} "
                    "does not have a filter."
                )

            match_elements, match_obs = self.apply_filter_single(tpm)
            if not match_obs.any():
                if not self.allow_empty:
                    # If empty results are not allowed, we raise an error
                    raise ValueError(
                        f"The TwoPointMeasurement {tpm} does not have any elements "
                        f"matching the filter."
                    )
                # If the filter is empty, we skip this measurement
                continue

            assert isinstance(tpm.metadata, (TwoPointReal, TwoPointHarmonic))
            new_metadata: TwoPointReal | TwoPointHarmonic
            match tpm.metadata:
                case TwoPointReal():
                    new_metadata = TwoPointReal(
                        XY=tpm.metadata.XY,
                        thetas=tpm.metadata.thetas[match_elements],
                    )
                case TwoPointHarmonic():
                    # If the window function is not None, we need to filter it as well
                    # and update the metadata accordingly.
                    new_metadata = TwoPointHarmonic(
                        XY=tpm.metadata.XY,
                        window=(
                            tpm.metadata.window[:, match_obs][match_elements, :]
                            if tpm.metadata.window is not None
                            else None
                        ),
                        ells=tpm.metadata.ells[match_elements],
                    )

            result.append(
                TwoPointMeasurement(
                    data=tpm.data[match_obs],
                    indices=tpm.indices[match_obs],
                    covariance_name=tpm.covariance_name,
                    metadata=new_metadata,
                )
            )

        return result
