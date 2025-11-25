"""Type definitions and Pydantic models for data_functions module."""

from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)

from firecrown.metadata_functions import make_measurement, make_measurement_dict
from firecrown.metadata_types import (
    Measurement,
    TwoPointFilterMethod,
    TwoPointHarmonic,
    TwoPointReal,
)


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


BinSpec = frozenset[TwoPointTracerSpec]


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


def bin_spec_from_metadata(metadata: TwoPointReal | TwoPointHarmonic):
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
