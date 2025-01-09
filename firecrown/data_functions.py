"""This module deals with two-point data functions.

It contains functions to manipulate two-point data objects.
"""

import hashlib
from typing import Callable, Sequence, Annotated

import sacc
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
    PrivateAttr,
    field_serializer,
)
import numpy as np
import numpy.typing as npt

from firecrown.metadata_types import (
    TwoPointHarmonic,
    TwoPointReal,
    Measurement,
)
from firecrown.metadata_functions import (
    extract_all_tracers_inferred_galaxy_zdists,
    extract_window_function,
    extract_all_harmonic_metadata_indices,
    extract_all_real_metadata_indices,
    make_two_point_xy,
    make_measurement,
    make_measurement_dict,
)
from firecrown.data_types import TwoPointMeasurement


def cov_hash(sacc_data: sacc.Sacc) -> str:
    """Return a hash of the covariance matrix.

    :param sacc_data: The SACC data object containing the covariance matrix.
    :return: The hash of the covariance matrix.
    """
    return hashlib.sha256(sacc_data.covariance.dense).hexdigest()


def extract_all_harmonic_data(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
    include_maybe_types=False,
) -> list[TwoPointMeasurement]:
    """Extract the two-point function metadata and data from a sacc file."""
    if sacc_data.covariance is None or sacc_data.covariance.dense is None:
        raise ValueError("The SACC object does not have a dense covariance matrix.")

    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, include_maybe_types=include_maybe_types
        )
    }

    result: list[TwoPointMeasurement] = []
    for cell_index in extract_all_harmonic_metadata_indices(
        sacc_data, allowed_data_type
    ):
        t1, t2 = cell_index["tracer_names"]
        dt = cell_index["data_type"]

        ells, Cells, indices = sacc_data.get_ell_cl(
            data_type=dt, tracer1=t1, tracer2=t2, return_cov=False, return_ind=True
        )

        ells, weights = maybe_enforce_window(ells, indices, sacc_data)

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
                    ells=ells,
                ),
            ),
        )

    return result


def maybe_enforce_window(
    ells: npt.NDArray[np.int64], indices: npt.NDArray[np.int64], sacc_data: sacc.Sacc
) -> tuple[npt.NDArray[np.int64], None | npt.NDArray[np.float64]]:
    """Possibly enforce a window function on the given ells.

    :param ells: The original ell values.
    :param indices: The indices of the data points in the SACC object.
    :param sacc_data: The SACC object containing the data.
    :return: A tuple containing the possibly replaced ells and the window weights.
    """
    replacement_ells, weights = extract_window_function(sacc_data, indices)
    if replacement_ells is not None:
        ells = replacement_ells
    return ells, weights


# Extracting the two-point function metadata and data from a sacc file


def extract_all_real_data(
    sacc_data: sacc.Sacc,
    allowed_data_type: None | list[str] = None,
    include_maybe_types=False,
) -> list[TwoPointMeasurement]:
    """Extract the two-point function metadata and data from a sacc file."""
    if sacc_data.covariance is None or sacc_data.covariance.dense is None:
        raise ValueError("The SACC object does not have a dense covariance matrix.")

    inferred_galaxy_zdists_dict = {
        igz.bin_name: igz
        for igz in extract_all_tracers_inferred_galaxy_zdists(
            sacc_data, include_maybe_types=include_maybe_types
        )
    }

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
        index_set: set[int] = set(measurement.indices)
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


class TwoPointTracerSpec(BaseModel):
    """Class defining a tracer bin specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bin_name: Annotated[str, Field(description="The name of the tracer bin.")]
    bin_measurement: Annotated[
        Measurement,
        Field(description="The measurement of the tracer bin."),
        BeforeValidator(make_measurement),
    ]

    @field_serializer("bin_measurement")
    @classmethod
    def serialize_bin_measurement(cls, value: Measurement) -> dict[str, str]:
        """Serialize the Measurement."""
        return make_measurement_dict(value)


def make_interval_from_list(
    values: list[float] | tuple[float, float]
) -> tuple[float, float]:
    """Create an interval from a list of values."""
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError("The list should have two values.")
        if not all(isinstance(v, float) for v in values):
            raise ValueError("The list should have two float values.")

        return (values[0], values[1])
    if isinstance(values, tuple):
        return values

    raise ValueError("The values should be a list or a tuple.")


class TwoPointBinFilter(BaseModel):
    """Class defining a filter for a bin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bin_spec: Annotated[
        list[TwoPointTracerSpec],
        Field(
            description="The two-point bin specification.",
        ),
    ]
    bin_filter: Annotated[
        tuple[float, float],
        BeforeValidator(make_interval_from_list),
        Field(description="The range of the bin to filter."),
    ]

    @model_validator(mode="after")
    def check_bin_filter(self) -> "TwoPointBinFilter":
        """Check the bin filter."""
        if self.bin_filter[0] >= self.bin_filter[1]:
            raise ValueError("The bin filter should be a valid range.")
        if 1 > len(self.bin_spec) > 2:
            raise ValueError("The bin_spec must contain one or two elements.")
        return self

    @field_serializer("bin_filter")
    @classmethod
    def serialize_bin_filter(cls, value: tuple[float, float]) -> list[float]:
        """Serialize the Measurement."""
        return list(value)

    @classmethod
    def from_args(
        cls,
        bin_name1: str,
        bin_measurement1: Measurement,
        bin_name2: str,
        bin_measurement2: Measurement,
        bin_lower: float,
        bin_upper: float,
    ) -> "TwoPointBinFilter":
        """Create a TwoPointBinFilter from the arguments."""
        return cls(
            bin_spec=[
                TwoPointTracerSpec(
                    bin_name=bin_name1, bin_measurement=bin_measurement1
                ),
                TwoPointTracerSpec(
                    bin_name=bin_name2, bin_measurement=bin_measurement2
                ),
            ],
            bin_filter=(bin_lower, bin_upper),
        )

    @classmethod
    def from_args_auto(
        cls,
        bin_name: str,
        bin_measurement: Measurement,
        bin_lower: float,
        bin_upper: float,
    ) -> "TwoPointBinFilter":
        """Create a TwoPointBinFilter from the arguments."""
        return cls(
            bin_spec=[
                TwoPointTracerSpec(bin_name=bin_name, bin_measurement=bin_measurement),
            ],
            bin_filter=(bin_lower, bin_upper),
        )


BinSpec = frozenset[TwoPointTracerSpec]


def bin_spec_from_metadata(metadata: TwoPointReal | TwoPointHarmonic) -> BinSpec:
    """Return the bin spec from the metadata."""
    return frozenset(
        (
            TwoPointTracerSpec(
                bin_name=metadata.XY.x.bin_name,
                bin_measurement=metadata.XY.x_measurement,
            ),
            TwoPointTracerSpec(
                bin_name=metadata.XY.y.bin_name,
                bin_measurement=metadata.XY.y_measurement,
            ),
        )
    )


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
    allow_empty: bool = Field(
        default=False,
        description=(
            "When true, objects with no elements remaining after applying "
            "the filter will be ignored rather than treated as an error."
        ),
    )

    _bin_filter_dict: dict[BinSpec, tuple[float, float]] = PrivateAttr()

    @model_validator(mode="after")
    def check_bin_filters(self) -> "TwoPointBinFilterCollection":
        """Check the bin filters."""
        bin_specs = set()
        for bin_filter in self.bin_filters:
            bin_spec = frozenset(bin_filter.bin_spec)
            if bin_spec in bin_specs:
                raise ValueError(
                    f"The bin name {bin_filter.bin_spec} is repeated "
                    f"in the bin filters."
                )
            bin_specs.add(bin_spec)

        self._bin_filter_dict = {
            frozenset(bin_filter.bin_spec): bin_filter.bin_filter
            for bin_filter in self.bin_filters
        }
        return self

    @property
    def bin_filter_dict(self) -> dict[BinSpec, tuple[float, float]]:
        """Return the bin filter dictionary."""
        return self._bin_filter_dict

    def filter_match(self, tpm: TwoPointMeasurement) -> bool:
        """Check if the TwoPointMeasurement matches the filter."""
        bin_spec_key = bin_spec_from_metadata(tpm.metadata)
        if bin_spec_key not in self._bin_filter_dict:
            return False
        return True

    def run_bin_filter(
        self,
        bin_filter: tuple[float, float],
        vals: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    ) -> npt.NDArray[np.bool_]:
        """Run the filter merge."""
        return (vals >= bin_filter[0]) & (vals <= bin_filter[1])

    def apply_filter_single(
        self, tpm: TwoPointMeasurement
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Apply the filter to a single TwoPointMeasurement."""
        assert self.filter_match(tpm)
        bin_spec_key = bin_spec_from_metadata(tpm.metadata)
        bin_filter = self._bin_filter_dict[bin_spec_key]
        if tpm.is_real():
            assert isinstance(tpm.metadata, TwoPointReal)
            match_elements = self.run_bin_filter(bin_filter, tpm.metadata.thetas)
            return match_elements, match_elements

        assert isinstance(tpm.metadata, TwoPointHarmonic)
        match_elements = self.run_bin_filter(bin_filter, tpm.metadata.ells)
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
                        f"The TwoPointMeasurement {tpm.metadata} does not "
                        f"have any elements matching the filter."
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
