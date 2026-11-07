"""Binned cluster measurement data type."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from firecrown.metadata_types import BinnedClusterMetadata
from firecrown.utils import YAMLSerializable


@dataclass(frozen=True, kw_only=True)
class BinnedClusterMeasurement(YAMLSerializable):
    """The data of one binned cluster observable, with its metadata.

    :ivar data: The measured values, one per bin.
    :ivar indices: The indices of the data points in the SACC data vector.
    :ivar covariance_name: The name of the covariance matrix.
    :ivar metadata: The observable, survey and bins.
    """

    data: npt.NDArray[np.float64]
    indices: npt.NDArray[np.int64]
    covariance_name: str
    metadata: BinnedClusterMetadata

    def __post_init__(self) -> None:
        """Check that data, indices and bins have the same length."""
        if len(self.data.shape) != 1:
            raise ValueError("Data should be a 1D array.")
        if self.data.shape != self.indices.shape:
            raise ValueError("Data and indices should have the same shape.")
        if len(self.data) != self.metadata.n_observations():
            raise ValueError("Data and metadata should have the same length.")

    def __eq__(self, other: object) -> bool:
        """Equality test for BinnedClusterMeasurement objects."""
        if not isinstance(other, BinnedClusterMeasurement):
            return NotImplemented
        return (
            np.array_equal(self.data, other.data)
            and np.array_equal(self.indices, other.indices)
            and self.covariance_name == other.covariance_name
            and self.metadata == other.metadata
        )
