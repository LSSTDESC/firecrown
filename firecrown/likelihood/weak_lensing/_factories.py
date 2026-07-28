"""Factory classes for weak lensing sources and systematics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from firecrown.likelihood.weak_lensing._args import WeakLensingArgs
from firecrown.likelihood.weak_lensing._source import WeakLensing
from firecrown.likelihood.weak_lensing._systematics import (
    LinearAlignmentSystematic,
    MultiplicativeShearBias,
    TattAlignmentSystematic,
)
from firecrown.likelihood_base import (
    PhotoZShiftandStretchFactory,
    PhotoZShiftFactory,
    SourceGalaxySystematic,
)
from firecrown.metadata_types import ProjectedField, TomographicBin, TypeSource


class MultiplicativeShearBiasFactory(BaseModel):
    """Factory class for MultiplicativeShearBias objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["MultiplicativeShearBiasFactory"],
        Field(description="The type of the systematic."),
    ] = "MultiplicativeShearBiasFactory"

    def create(self, bin_name: str) -> MultiplicativeShearBias:
        """Create a MultiplicativeShearBias object.

        :param tomographic_bin: The inferred galaxy redshift distribution for
            the created MultiplicativeShearBias object.
        :return: The created MultiplicativeShearBias object.
        """
        return MultiplicativeShearBias(bin_name)

    def create_global(self) -> MultiplicativeShearBias:
        """Create a MultiplicativeShearBias object.

        :return: The created MultiplicativeShearBias object.
        """
        raise ValueError("MultiplicativeShearBias cannot be global")


class LinearAlignmentSystematicFactory(BaseModel):
    """Factory class for LinearAlignmentSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["LinearAlignmentSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "LinearAlignmentSystematicFactory"

    alphag: None | float = 1.0

    def create(self, bin_name: str) -> LinearAlignmentSystematic:
        """Create a LinearAlignmentSystematic object.

        :param tomographic_bin: The inferred galaxy redshift distribution for
            the created LinearAlignmentSystematic object.
        :return: The created LinearAlignmentSystematic object.
        """
        return LinearAlignmentSystematic(bin_name)

    def create_global(self) -> LinearAlignmentSystematic:
        """Create a LinearAlignmentSystematic object.

        :return: The created LinearAlignmentSystematic object.
        """
        return LinearAlignmentSystematic(sacc_tracer=None, alphag=self.alphag)


class TattAlignmentSystematicFactory(BaseModel):
    """Factory class for TattAlignmentSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["TattAlignmentSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "TattAlignmentSystematicFactory"
    include_z_dependence: bool = False

    def create(self, bin_name: str) -> TattAlignmentSystematic:
        """Create a TattAlignmentSystematic object.

        :param tomographic_bin: The inferred galaxy redshift distribution for
            the created TattAlignmentSystematic object.
        :return: The created TattAlignmentSystematic object.
        """
        return TattAlignmentSystematic(bin_name, self.include_z_dependence)

    def create_global(self) -> TattAlignmentSystematic:
        """Create a TattAlignmentSystematic object.

        :return: The created TattAlignmentSystematic object.
        """
        return TattAlignmentSystematic(None, self.include_z_dependence)


WeakLensingSystematicFactory = Annotated[
    PhotoZShiftFactory
    | PhotoZShiftandStretchFactory
    | MultiplicativeShearBiasFactory
    | LinearAlignmentSystematicFactory
    | TattAlignmentSystematicFactory,
    Field(discriminator="type", union_mode="left_to_right"),
]


class WeakLensingFactory(BaseModel):
    """Factory class for WeakLensing objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    _cache: dict[int, WeakLensing] = PrivateAttr()
    _global_systematics_instances: Sequence[SourceGalaxySystematic[WeakLensingArgs]] = (
        PrivateAttr()
    )

    type_source: TypeSource = TypeSource.DEFAULT
    per_bin_systematics: Sequence[WeakLensingSystematicFactory] = Field(
        default_factory=list
    )
    global_systematics: Sequence[WeakLensingSystematicFactory] = Field(
        default_factory=list
    )

    def model_post_init(self, _, /) -> None:
        """Initialize the WeakLensingFactory object."""
        self._cache: dict[int, WeakLensing] = {}
        self._global_systematics_instances = [
            wl_systematic_factory.create_global()
            for wl_systematic_factory in self.global_systematics
        ]

    def create(self, tomographic_bin: ProjectedField) -> WeakLensing:
        """Create a WeakLensing object with the given tracer name and scale."""
        assert isinstance(tomographic_bin, TomographicBin)
        inferred_zdist_id = id(tomographic_bin)
        if inferred_zdist_id in self._cache:
            return self._cache[inferred_zdist_id]

        systematics: list[SourceGalaxySystematic[WeakLensingArgs]] = [
            systematic_factory.create(tomographic_bin.bin_name)
            for systematic_factory in self.per_bin_systematics
        ]
        systematics.extend(self._global_systematics_instances)

        wl = WeakLensing.create_ready(tomographic_bin, systematics)
        self._cache[inferred_zdist_id] = wl

        return wl

    def create_from_metadata_only(
        self,
        sacc_tracer: str,
    ) -> WeakLensing:
        """Create an WeakLensing object with the given tracer name and scale."""
        sacc_tracer_id = hash(sacc_tracer)  # Improve this
        if sacc_tracer_id in self._cache:
            return self._cache[sacc_tracer_id]
        systematics: list[SourceGalaxySystematic[WeakLensingArgs]] = [
            systematic_factory.create(sacc_tracer)
            for systematic_factory in self.per_bin_systematics
        ]
        systematics.extend(self._global_systematics_instances)

        wl = WeakLensing(sacc_tracer=sacc_tracer, systematics=systematics)
        self._cache[sacc_tracer_id] = wl

        return wl
