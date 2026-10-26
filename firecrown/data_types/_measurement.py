"""Two-point measurement data type.

This module contains the TwoPointMeasurement class for storing two-point function
measurements on a sphere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from firecrown._integer_arrays import _IntBit, _as_int64
from firecrown._source_arrays import _NBit, _RealArray, _as_float64
from firecrown.metadata_types import TwoPointHarmonic, TwoPointReal
from firecrown.utils import YAMLSerializable


@dataclass(frozen=True, kw_only=True, init=False)
class TwoPointMeasurement(YAMLSerializable):
    """Class defining the data for a two-point measurement.

    The class used to store the data for a two-point function measured on a sphere.

    This includes the measured two-point function, their indices in the covariance
    matrix and the name of the covariance matrix. The corresponding metadata is also
    stored.
    """

    data: npt.NDArray[np.float64]
    indices: npt.NDArray[np.int64]
    covariance_name: str
    metadata: TwoPointReal | TwoPointHarmonic

    def __init__(
        self,
        *,
        data: _RealArray[_NBit],
        indices: npt.NDArray[np.integer[_IntBit]],
        covariance_name: str,
        metadata: TwoPointReal | TwoPointHarmonic,
    ) -> None:
        """Store float64 values and checked int64 covariance indices.

        :param data: Real integer or floating measured two-point values; conforming
            arrays are shared. Boolean, complex, string, object and masked arrays
            are rejected.
        :param indices: Integer positions in the covariance matrix.
        :param covariance_name: Name of the associated covariance matrix.
        :param metadata: Real or harmonic measurement metadata.
        """
        object.__setattr__(self, "data", _as_float64(data))
        object.__setattr__(self, "indices", _as_int64(indices))
        object.__setattr__(self, "covariance_name", covariance_name)
        object.__setattr__(self, "metadata", metadata)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Make sure the data and indices have the same shape."""
        if len(self.data.shape) != 1:
            raise ValueError("Data should be a 1D array.")

        if self.data.shape != self.indices.shape:
            raise ValueError("Data and indices should have the same shape.")

        if not isinstance(self.metadata, (TwoPointReal, TwoPointHarmonic)):
            raise ValueError(
                "Metadata should be an instance of TwoPointReal or TwoPointHarmonic."
            )

        if len(self.data) != self.metadata.n_observations():
            raise ValueError("Data and metadata should have the same length.")

    def __eq__(self, other: object) -> bool:
        """Equality test for TwoPointMeasurement objects."""
        if not isinstance(other, TwoPointMeasurement):
            return NotImplemented
        return (
            np.array_equal(self.data, other.data)
            and np.array_equal(self.indices, other.indices)
            and self.covariance_name == other.covariance_name
            and self.metadata == other.metadata
        )

    def is_real(self) -> bool:
        """Check if the metadata is real."""
        return isinstance(self.metadata, TwoPointReal)

    def is_harmonic(self) -> bool:
        """Check if the metadata is harmonic."""
        return isinstance(self.metadata, TwoPointHarmonic)
