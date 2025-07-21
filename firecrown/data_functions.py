"""This module deals with two-point data functions.

It contains functions to manipulate two-point data objects.
"""

import hashlib
from typing import Callable, Sequence, Annotated
from typing_extensions import assert_never

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
    TwoPointFilterMethod,
)
from firecrown.metadata_functions import (
    extract_all_tracers_inferred_galaxy_zdists,
    maybe_enforce_window,
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
    measurement: str,
    index_set: set[int],
    index_sets: list[set[int]],
    other_measurements: list[str],
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

        measurements_names = [f"{m}" for m in measurements]
        if seen_indices & index_set:
            ensure_no_overlaps(
                f"{measurement}", index_set, index_sets, measurements_names
            )
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

    name: Annotated[str, Field(description="The name of the tracer bin.")]
    measurement: Annotated[
        Measurement,
        Field(description="The measurement of the tracer bin."),
        BeforeValidator(make_measurement),
    ]

    @field_serializer("measurement")
    @classmethod
    def serialize_measurement(cls, value: Measurement) -> dict[str, str]:
        """Serialize the Measurement."""
        return make_measurement_dict(value)


def make_interval_from_list(
    values: list[float] | tuple[float, float],
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
    """Class defining a filter for a bin.

    :param spec: The two-point bin specification.
    :param interval: The range of the bin to filter.
    :param method: The filter method.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec: Annotated[
        list[TwoPointTracerSpec],
        Field(
            description="The two-point bin specification.",
        ),
    ]
    interval: Annotated[
        tuple[float, float],
        BeforeValidator(make_interval_from_list),
        Field(description="The range of the bin to filter."),
    ]
    method: Annotated[TwoPointFilterMethod, Field(description="The filter method.")] = (
        TwoPointFilterMethod.SUPPORT
    )

    @model_validator(mode="after")
    def check_bin_filter(self) -> "TwoPointBinFilter":
        """Check the bin filter."""
        if self.interval[0] >= self.interval[1]:
            raise ValueError("The bin filter should be a valid range.")
        if not 1 <= len(self.spec) <= 2:
            raise ValueError("The bin_spec must contain one or two elements.")
        return self

    @field_serializer("interval")
    @classmethod
    def serialize_interval(cls, value: tuple[float, float]) -> list[float]:
        """Serialize the Measurement."""
        return list(value)

    @classmethod
    def from_args(
        cls,
        name1: str,
        measurement1: Measurement,
        name2: str,
        measurement2: Measurement,
        lower: float,
        upper: float,
        method: TwoPointFilterMethod = TwoPointFilterMethod.SUPPORT,
    ) -> "TwoPointBinFilter":
        """Create a TwoPointBinFilter from the arguments."""
        return cls(
            spec=[
                TwoPointTracerSpec(name=name1, measurement=measurement1),
                TwoPointTracerSpec(name=name2, measurement=measurement2),
            ],
            interval=(lower, upper),
            method=method,
        )

    @classmethod
    def from_args_auto(
        cls,
        name: str,
        measurement: Measurement,
        lower: float,
        upper: float,
        method: TwoPointFilterMethod = TwoPointFilterMethod.SUPPORT,
    ) -> "TwoPointBinFilter":
        """Create a TwoPointBinFilter from the arguments."""
        return cls(
            spec=[
                TwoPointTracerSpec(name=name, measurement=measurement),
            ],
            interval=(lower, upper),
            method=method,
        )


BinSpec = frozenset[TwoPointTracerSpec]


def bin_spec_from_metadata(metadata: TwoPointReal | TwoPointHarmonic) -> BinSpec:
    """Return the bin spec from the metadata."""
    return frozenset(
        (
            TwoPointTracerSpec(
                name=metadata.XY.x.bin_name,
                measurement=metadata.XY.x_measurement,
            ),
            TwoPointTracerSpec(
                name=metadata.XY.y.bin_name,
                measurement=metadata.XY.y_measurement,
            ),
        )
    )


class TwoPointBinFilterCollection(BaseModel):
    """Class defining a collection of bin filters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    require_filter_for_all: bool = Field(
        default=False,
        description="If True, all bins should match a filter.",
    )
    allow_empty: bool = Field(
        default=False,
        description=(
            "When true, objects with no elements remaining after applying "
            "the filter will be ignored rather than treated as an error."
        ),
    )
    filters: list[TwoPointBinFilter] = Field(
        description="The list of bin filters.",
    )

    _bin_filter_dict: dict[BinSpec, TwoPointBinFilter] = PrivateAttr()

    @model_validator(mode="after")
    def check_bin_filters(self) -> "TwoPointBinFilterCollection":
        """Check the bin filters."""
        bin_specs = set()
        for bin_filter in self.filters:
            bin_spec = frozenset(bin_filter.spec)
            if bin_spec in bin_specs:
                raise ValueError(
                    f"The bin name {bin_filter.spec} is repeated "
                    f"in the bin filters."
                )
            bin_specs.add(bin_spec)

        self._bin_filter_dict = {
            frozenset(bin_filter.spec): bin_filter for bin_filter in self.filters
        }
        return self

    @property
    def bin_filter_dict(self) -> dict[BinSpec, TwoPointBinFilter]:
        """Return the bin filter dictionary."""
        return self._bin_filter_dict

    def filter_match(self, tpm: TwoPointMeasurement) -> bool:
        """Check if the TwoPointMeasurement matches the filter."""
        bin_spec_key = bin_spec_from_metadata(tpm.metadata)
        return bin_spec_key in self._bin_filter_dict

    def run_bin_filter(
        self,
        bin_filter: TwoPointBinFilter,
        vals: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    ) -> npt.NDArray[np.bool_]:
        """Run the filter merge."""
        return (vals >= bin_filter.interval[0]) & (vals <= bin_filter.interval[1])

    def _apply_filter_single_window(
        self, bin_filter: TwoPointBinFilter, tpm: TwoPointMeasurement
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Apply a filter to a TwoPointMeasurement with a window function.

        The window function is a matrix where each column contains weights over ell
        values for a given observation. The filtering process depends on the method:

        - If filtering by `LABEL`, `window_ells` are filtered directly.
        - Otherwise, the filter is applied to `ells`, producing a mask
          (`match_elements`).
        For each observation, the sum of weights inside this mask is computed.
        Observations are kept if their support lies within the allowed fraction:
            - `SUPPORT`: full support (sum < 1.0)
            - `SUPPORT_95`: >=95% of support must be inside the filtered ell range
        """
        assert isinstance(tpm.metadata, TwoPointHarmonic)

        match bin_filter.method:
            case TwoPointFilterMethod.LABEL:
                assert tpm.metadata.window_ells is not None
                match_obs = self.run_bin_filter(bin_filter, tpm.metadata.window_ells)
                match_elements = np.ones_like(tpm.metadata.ells, dtype=bool)
                return match_elements, match_obs
            case TwoPointFilterMethod.SUPPORT:
                assert tpm.metadata.window is not None
                match_elements = self.run_bin_filter(bin_filter, tpm.metadata.ells)
                support = tpm.metadata.window[match_elements].sum(axis=0)
                match_obs = support >= 0.999
                return match_elements, match_obs
            case TwoPointFilterMethod.SUPPORT_95:
                assert tpm.metadata.window is not None
                match_elements = self.run_bin_filter(bin_filter, tpm.metadata.ells)
                support = tpm.metadata.window[match_elements].sum(axis=0)
                match_obs = support >= 0.95
                return match_elements, match_obs
            case _ as unreachable:
                assert_never(unreachable)

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
        if tpm.metadata.window is None:
            match_elements = self.run_bin_filter(bin_filter, tpm.metadata.ells)
            return match_elements, match_elements

        return self._apply_filter_single_window(bin_filter, tpm)

    def __call__(
        self, tpms: Sequence[TwoPointMeasurement]
    ) -> list[TwoPointMeasurement]:
        """Filter the two-point measurements."""
        result = []

        for tpm in tpms:
            if not self.filter_match(tpm):
                if not self.require_filter_for_all:
                    result.append(tpm)
                    continue
                raise ValueError(f"The bin name {tpm.metadata} does not have a filter.")

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
                        window_ells=(
                            tpm.metadata.window_ells[match_obs]
                            if tpm.metadata.window_ells is not None
                            else None
                        ),
                        ells=tpm.metadata.ells[match_elements],
                    )
                case _ as unreachable:
                    assert_never(unreachable)

            result.append(
                TwoPointMeasurement(
                    data=tpm.data[match_obs],
                    indices=tpm.indices[match_obs],
                    covariance_name=tpm.covariance_name,
                    metadata=new_metadata,
                )
            )

        return result
