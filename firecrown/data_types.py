"""This module deals with data types.

This module contains data types definitions.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from firecrown.utils import YAMLSerializable
from firecrown.metadata_types import TwoPointReal, TwoPointHarmonic


@dataclass(frozen=True, kw_only=True)
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

    def __eq__(self, other) -> bool:
        """Equality test for TwoPointMeasurement objects."""
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
