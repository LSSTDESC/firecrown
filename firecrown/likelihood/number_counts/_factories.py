"""Factory classes for number counts sources and systematics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from firecrown.likelihood.number_counts._source import NumberCounts
from firecrown.likelihood.number_counts._systematics import (
    ConstantMagnificationBiasSystematic,
    LinearBiasSystematic,
    MagnificationBiasSystematic,
    PTNonLinearBiasSystematic,
)
from firecrown.likelihood._source import SpecZStretchFactory
from firecrown.likelihood.weak_lensing import (
    PhotoZShiftandStretchFactory,
    PhotoZShiftFactory,
)
from firecrown.likelihood._base import SourceGalaxySystematic
from firecrown.likelihood.number_counts._args import NumberCountsArgs
from firecrown.metadata_types import InferredGalaxyZDist, TypeSource


class LinearBiasSystematicFactory(BaseModel):
    """Factory class for LinearBiasSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["LinearBiasSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "LinearBiasSystematicFactory"

    def create(self, bin_name: str) -> LinearBiasSystematic:
        """Create a LinearBiasSystematic object with the given tracer name."""
        return LinearBiasSystematic(bin_name)

    def create_global(self) -> LinearBiasSystematic:
        """Create a LinearBiasSystematic object with the given tracer name."""
        raise ValueError("LinearBiasSystematic cannot be global.")


class PTNonLinearBiasSystematicFactory(BaseModel):
    """Factory class for PTNonLinearBiasSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["PTNonLinearBiasSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "PTNonLinearBiasSystematicFactory"

    def create(self, bin_name: str) -> PTNonLinearBiasSystematic:
        """Create a PTNonLinearBiasSystematic object with the given tracer name.

        :param bin_name: the name of the bin
        :return: the created PTNonLinearBiasSystematic object
        """
        return PTNonLinearBiasSystematic(bin_name)

    def create_global(self) -> PTNonLinearBiasSystematic:
        """Create a global PTNonLinearBiasSystematic object."""
        return PTNonLinearBiasSystematic()


class MagnificationBiasSystematicFactory(BaseModel):
    """Factory class for MagnificationBiasSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["MagnificationBiasSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "MagnificationBiasSystematicFactory"

    def create(self, bin_name: str) -> MagnificationBiasSystematic:
        """Create a MagnificationBiasSystematic object with the given tracer name.

        :param bin_name: the name of the bin
        :return: the created MagnificationBiasSystematic object
        """
        return MagnificationBiasSystematic(bin_name)

    def create_global(self) -> MagnificationBiasSystematic:
        """Required by the interface, but raises an error.

        MagnificationBiasSystematic systematics cannot be global.
        """
        raise ValueError("MagnificationBiasSystematic cannot be global.")


class ConstantMagnificationBiasSystematicFactory(BaseModel):
    """Factory class for ConstantMagnificationBiasSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["ConstantMagnificationBiasSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "ConstantMagnificationBiasSystematicFactory"

    def create(self, bin_name: str) -> ConstantMagnificationBiasSystematic:
        """Create a ConstantMagnificationBiasSystematic object.

        :param bin_name: the name of the bin
        :return: the created ConstantMagnificationBiasSystematic object
        """
        return ConstantMagnificationBiasSystematic(bin_name)

    def create_global(self) -> ConstantMagnificationBiasSystematic:
        """Required by the interface, but raises an error.

        ConstantMagnificationBiasSystematic systematics cannot be global.
        """
        raise ValueError("ConstantMagnificationBiasSystematic cannot be global.")


NumberCountsSystematicFactory = Annotated[
    PhotoZShiftFactory
    | PhotoZShiftandStretchFactory
    | LinearBiasSystematicFactory
    | PTNonLinearBiasSystematicFactory
    | MagnificationBiasSystematicFactory
    | ConstantMagnificationBiasSystematicFactory
    | SpecZStretchFactory,
    Field(discriminator="type", union_mode="left_to_right"),
]


class NumberCountsFactory(BaseModel):
    """Factory class for NumberCounts objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    _cache: dict[int, NumberCounts] = PrivateAttr()
    _global_systematics_instances: Sequence[
        SourceGalaxySystematic[NumberCountsArgs]
    ] = PrivateAttr()

    type_source: TypeSource = TypeSource.DEFAULT
    per_bin_systematics: Sequence[NumberCountsSystematicFactory] = Field(
        default_factory=list
    )
    global_systematics: Sequence[NumberCountsSystematicFactory] = Field(
        default_factory=list
    )
    include_rsd: bool = False

    def model_post_init(self, _, /) -> None:
        """Initialize the NumberCountsFactory."""
        self._cache: dict[int, NumberCounts] = {}
        self._global_systematics_instances = [
            nc_systematic_factory.create_global()
            for nc_systematic_factory in self.global_systematics
        ]

    def create(self, inferred_zdist: InferredGalaxyZDist) -> NumberCounts:
        """Create a NumberCounts object with the given tracer name and scale.

        :param inferred_zdist: the inferred redshift distribution
        :return: a fully initialized NumberCounts object
        """
        inferred_zdist_id = id(inferred_zdist)
        if inferred_zdist_id in self._cache:
            return self._cache[inferred_zdist_id]

        systematics: list[SourceGalaxySystematic[NumberCountsArgs]] = [
            systematic_factory.create(inferred_zdist.bin_name)
            for systematic_factory in self.per_bin_systematics
        ]
        systematics.extend(self._global_systematics_instances)

        nc = NumberCounts.create_ready(
            inferred_zdist, systematics=systematics, has_rsd=self.include_rsd
        )
        self._cache[inferred_zdist_id] = nc

        return nc

    def create_from_metadata_only(
        self,
        sacc_tracer: str,
    ) -> NumberCounts:
        """Create an WeakLensing object with the given tracer name and scale.

        :param sacc_tracer: the name of the tracer
        :return: a fully initialized NumberCounts object
        """
        sacc_tracer_id = hash(sacc_tracer)  # Improve this
        if sacc_tracer_id in self._cache:
            return self._cache[sacc_tracer_id]
        systematics: list[SourceGalaxySystematic[NumberCountsArgs]] = [
            systematic_factory.create(sacc_tracer)
            for systematic_factory in self.per_bin_systematics
        ]
        systematics.extend(self._global_systematics_instances)

        nc = NumberCounts(
            sacc_tracer=sacc_tracer, systematics=systematics, has_rsd=self.include_rsd
        )
        self._cache[sacc_tracer_id] = nc

        return nc
