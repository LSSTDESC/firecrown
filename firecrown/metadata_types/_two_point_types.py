"""Two-point correlation types and metadata."""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic_core import core_schema

from firecrown.metadata_types._compatibility import measurement_is_compatible
from firecrown.metadata_types._inferred_galaxy_zdist import Tracer
from firecrown.metadata_types._measurements import (
    HARMONIC_ONLY_MEASUREMENTS,
    REAL_ONLY_MEASUREMENTS,
    Measurement,
)
from firecrown.metadata_types._sacc_type_string import (
    _type_to_sacc_string_harmonic,
    _type_to_sacc_string_real,
)
from firecrown.metadata_types._utils import TracerNames
from firecrown.utils import YAMLSerializable, compare_optional_arrays


def _measurement_supports_real(x: Measurement) -> bool:
    """Return True if x supports real-space calculations."""
    return x not in HARMONIC_ONLY_MEASUREMENTS


def _measurement_supports_harmonic(x: Measurement) -> bool:
    """Return True if x supports harmonic-space calculations."""
    return x not in REAL_ONLY_MEASUREMENTS


@dataclass(frozen=True, kw_only=True)
class TwoPointXY(YAMLSerializable):
    """Class defining a two-point correlation pair of redshift resolutions.

    It is used to store the two redshift resolutions for the two bins being
    correlated. The measurements must follow the canonical SACC ordering:
    CMB < Clusters < Galaxies, and within each type, the ordering defined by
    the Measurement enum (e.g., for Galaxies: shape measurements before counts).
    """

    # Use the generic Tracer protocol so other tracer implementations can be used
    x: Tracer
    y: Tracer
    x_measurement: Measurement
    y_measurement: Measurement

    def __post_init__(self) -> None:
        """Validate that measurements are compatible and follow canonical ordering."""
        if self.x_measurement not in self.x.measurements:
            raise ValueError(
                f"Measurement {self.x_measurement} not in the measurements of "
                f"{self.x.bin_name}."
            )
        if self.y_measurement not in self.y.measurements:
            raise ValueError(
                f"Measurement {self.y_measurement} not in the measurements of "
                f"{self.y.bin_name}."
            )
        if not measurement_is_compatible(self.x_measurement, self.y_measurement):
            raise ValueError(
                f"Measurements {self.x_measurement} and {self.y_measurement} "
                f"are not compatible."
            )

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointXY objects."""
        return (
            self.x == other.x
            and self.y == other.y
            and self.x_measurement == other.x_measurement
            and self.y_measurement == other.y_measurement
        )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointXY object."""
        return f"({self.x.bin_name}, {self.y.bin_name})"

    def get_tracer_names(self) -> TracerNames:
        """Return the TracerNames object for the TwoPointXY object."""
        return TracerNames(self.x.bin_name, self.y.bin_name)


@dataclass(frozen=True, kw_only=True)
class TwoPointHarmonic(YAMLSerializable):
    """Class defining the metadata for an harmonic-space two-point measurement.

    The class used to store the metadata for a (spherical) harmonic-space two-point
    function measured on a sphere.

    This includes the two redshift resolutions (one for each binned quantity) and the
    array of (integer) l's at which the two-point function which has this metadata were
    calculated.
    """

    XY: TwoPointXY
    ells: npt.NDArray[np.int64]
    window: None | npt.NDArray[np.float64] = None
    window_ells: None | npt.NDArray[np.float64] = None

    def __post_init__(self) -> None:
        """Validate the TwoPointHarmonic data.

        Make sure the ells are a 1D array and X and Y are compatible
        with harmonic-space calculations.
        """
        if len(self.ells.shape) != 1:
            raise ValueError("Ells should be a 1D array.")

        self._check_window_consistency()

        if not _measurement_supports_harmonic(
            self.XY.x_measurement
        ) or not _measurement_supports_harmonic(self.XY.y_measurement):
            raise ValueError(
                f"Measurements {self.XY.x_measurement} and "
                f"{self.XY.y_measurement} must support harmonic-space calculations."
            )

    def _check_window_consistency(self) -> None:
        """Make sure the window is consistent with the ells."""
        if self.window is not None:
            if not isinstance(self.window, np.ndarray):
                raise ValueError("window should be a ndarray.")
            if len(self.window.shape) != 2:
                raise ValueError("window should be a 2D array.")
            if self.window.shape[0] != len(self.ells):
                raise ValueError("window should have the same number of rows as ells.")
            if self.window_ells is None:
                raise ValueError("window_ells must be set if window is set.")
            if len(self.window_ells.shape) != 1:
                raise ValueError("window_ells should be a 1D array.")
            if self.window_ells.shape[0] != self.window.shape[1]:
                raise ValueError(
                    "window_ells should have the same number of "
                    "elements as the columns of window."
                )
        elif self.window_ells is not None:
            raise ValueError("window_ells must be None if window is None.")

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointHarmonic objects."""
        if not isinstance(other, TwoPointHarmonic):
            raise ValueError("Can only compare TwoPointHarmonic objects.")

        return (
            self.XY == other.XY
            and np.array_equal(self.ells, other.ells)
            and compare_optional_arrays(self.window, other.window)
        )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointHarmonic object."""
        return f"{self.XY}[{self.get_sacc_name()}]"

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return _type_to_sacc_string_harmonic(
            self.XY.x_measurement, self.XY.y_measurement
        )

    def n_observations(self) -> int:
        """Return the number of observations described by these metadata.

        :return: The number of observations.
        """
        if self.window is None:
            return self.ells.shape[0]
        return self.window.shape[1]


@dataclass(frozen=True, kw_only=True)
class TwoPointReal(YAMLSerializable):
    """Class defining the metadata for a real-space two-point measurement.

    The class used to store the metadata for a real-space two-point function measured
    on a sphere.

    This includes the two redshift resolutions (one for each binned quantity) and the a
    array of (floating point) theta (angle) values at which the two-point function
    which has  this metadata were calculated.
    """

    XY: TwoPointXY
    thetas: npt.NDArray[np.float64]

    def __post_init__(self):
        """Validate the TwoPointReal data.

        Make sure the window is
        """
        if len(self.thetas.shape) != 1:
            raise ValueError("Thetas should be a 1D array.")

        if not _measurement_supports_real(
            self.XY.x_measurement
        ) or not _measurement_supports_real(self.XY.y_measurement):
            raise ValueError(
                f"Measurements {self.XY.x_measurement} and "
                f"{self.XY.y_measurement} must support real-space calculations."
            )

    def __str__(self) -> str:
        """Return a string representation of the TwoPointReal object."""
        return f"{self.XY}[{self.get_sacc_name()}]"

    def get_sacc_name(self) -> str:
        """Return the SACC name for the two-point function."""
        return _type_to_sacc_string_real(self.XY.x_measurement, self.XY.y_measurement)

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointReal objects."""
        if not isinstance(other, TwoPointReal):
            raise ValueError("Can only compare TwoPointReal objects.")

        return self.XY == other.XY and np.array_equal(self.thetas, other.thetas)

    def n_observations(self) -> int:
        """Return the number of observations described by these metadata.

        :return: The number of observations.
        """
        return self.thetas.shape[0]


class TwoPointCorrelationSpace(YAMLSerializable, StrEnum):
    """This class defines the two-point correlation space.

    The two-point correlation space can be either real or harmonic. The real space
    corresponds measurements in terms of angular separation, while the harmonic space
    corresponds to measurements in terms of spherical harmonics decomposition.
    """

    REAL = auto()
    HARMONIC = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the TypeSource class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


class TwoPointFilterMethod(YAMLSerializable, StrEnum):
    """Defines methods for filtering two-point measurements.

    When filtering a two-point measurement with an associated window, the filter is
    applied to the window first. The user must then choose how to proceed:

    - If filtering by `LABEL`, the `window_ells` labels are used to determine whether
      the observation should be kept.
    - If filtering by `SUPPORT`, the full ell support must lie within the allowed range.
    - If filtering by `SUPPORT_95`, only the 95% support region must lie within the
      range.

    In all cases, filters define an interval of ells to retain.
    """

    LABEL = auto()
    SUPPORT = auto()
    SUPPORT_95 = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the TypeSource class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )
