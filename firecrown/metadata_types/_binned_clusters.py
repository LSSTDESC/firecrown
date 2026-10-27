"""Metadata types for binned cluster observables."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic_core import core_schema

from firecrown.utils import YAMLSerializable


class BinnedClusterObservable(YAMLSerializable, StrEnum):
    """Binned cluster observables; each value is its SACC data type."""

    COUNTS = "cluster_counts"
    MEAN_LOG_MASS = "cluster_mean_log_mass"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the BinnedClusterObservable class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


@dataclass(frozen=True, kw_only=True)
class ClusterSurvey(YAMLSerializable):
    """A cluster survey, read from a SACC SurveyTracer.

    :ivar name: The SACC tracer name.
    :ivar sky_area: The survey area in square degrees.
    """

    name: str
    sky_area: float

    def __post_init__(self) -> None:
        """Check that the name is not empty and the area is positive."""
        if self.name == "":
            raise ValueError("The survey name should not be empty.")
        if self.sky_area <= 0.0:
            raise ValueError("The survey sky_area should be positive.")


@dataclass(frozen=True, kw_only=True)
class ClusterBin(YAMLSerializable):
    """A redshift and richness bin, read from a SACC BinZTracer and BinRichnessTracer.

    The edges are the values stored in the SACC tracers.

    :ivar z_tracer: The BinZTracer name.
    :ivar z_edges: The lower and upper redshift edges.
    :ivar proxy_tracer: The BinRichnessTracer name.
    :ivar proxy_edges: The lower and upper richness edges.
    """

    z_tracer: str
    z_edges: tuple[float, float]
    proxy_tracer: str
    proxy_edges: tuple[float, float]

    def __post_init__(self) -> None:
        """Check that each pair of edges is increasing."""
        if not self.z_edges[0] < self.z_edges[1]:
            raise ValueError(f"The z_edges {self.z_edges} should be increasing.")
        if not self.proxy_edges[0] < self.proxy_edges[1]:
            raise ValueError(
                f"The proxy_edges {self.proxy_edges} should be increasing."
            )


@dataclass(frozen=True, kw_only=True)
class BinnedClusterMetadata(YAMLSerializable):
    """One binned cluster observable of one survey, with its bins in SACC order.

    :ivar survey: The survey.
    :ivar observable: The observable.
    :ivar bins: The bins, one per data point.
    """

    survey: ClusterSurvey
    observable: BinnedClusterObservable
    bins: tuple[ClusterBin, ...]

    def __post_init__(self) -> None:
        """Check that there is at least one bin and no bin is repeated."""
        if len(self.bins) == 0:
            raise ValueError("BinnedClusterMetadata should have at least one bin.")
        if len(set(self.bins)) != len(self.bins):
            raise ValueError("BinnedClusterMetadata should not repeat a bin.")

    def n_observations(self) -> int:
        """Return the number of data points, one per bin."""
        return len(self.bins)
