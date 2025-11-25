"""Number counts source class."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import final

import numpy as np
import pyccl

from firecrown import parameters
from firecrown.likelihood.number_counts._args import NumberCountsArgs
from firecrown.likelihood._base import (
    SourceGalaxy,
    SourceGalaxySystematic,
    Tracer,
)
from firecrown.metadata_types import InferredGalaxyZDist
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import (
    DerivedParameter,
    DerivedParameterCollection,
    ParamsMap,
)
from firecrown.updatable import UpdatableCollection


NUMBER_COUNTS_DEFAULT_BIAS = 1.5


class NumberCounts(SourceGalaxy[NumberCountsArgs]):
    """Source class for number counts."""

    def __init__(
        self,
        *,
        sacc_tracer: str,
        has_rsd: bool = False,
        derived_scale: bool = False,
        scale: float = 1.0,
        systematics: None | Sequence[SourceGalaxySystematic[NumberCountsArgs]] = None,
    ):
        """Initialize the NumberCounts object.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        :param has_rsd: whether to include RSD in the tracer.
        :param derived_scale: whether to include a derived parameter for the scale
            of the tracer.
        :param scale: the initial scale of the tracer.
        :param systematics: a list of systematics to apply to the tracer.
        """
        super().__init__(sacc_tracer=sacc_tracer, systematics=systematics)

        self.sacc_tracer = sacc_tracer
        self.has_rsd = has_rsd
        self.derived_scale = derived_scale

        self.bias = parameters.register_new_updatable_parameter(
            default_value=NUMBER_COUNTS_DEFAULT_BIAS
        )
        self.systematics: UpdatableCollection[
            SourceGalaxySystematic[NumberCountsArgs]
        ] = UpdatableCollection(systematics)
        self.scale = scale
        self.current_tracer_args: None | NumberCountsArgs = None
        self.tracer_args: NumberCountsArgs

    @classmethod
    def create_ready(
        cls,
        inferred_zdist: InferredGalaxyZDist,
        has_rsd: bool = False,
        derived_scale: bool = False,
        scale: float = 1.0,
        systematics: None | list[SourceGalaxySystematic[NumberCountsArgs]] = None,
    ) -> NumberCounts:
        """Create a NumberCounts object with the given tracer name and scale.

        This is the recommended way to create a NumberCounts object. It creates
        a fully initialized object.

        :param inferred_zdist: the inferred redshift distribution
        :param has_rsd: whether to include RSD in the tracer
        :param derived_scale: whether to include a derived parameter for the scale
            of the tracer
        :param scale: the initial scale of the tracer
        :param systematics: a list of systematics to apply to the tracer
        :return: a fully initialized NumberCounts object
        """
        obj = cls(
            sacc_tracer=inferred_zdist.bin_name,
            systematics=systematics,
            has_rsd=has_rsd,
            derived_scale=derived_scale,
            scale=scale,
        )
        # pylint: disable=unexpected-keyword-arg
        obj.tracer_args = NumberCountsArgs(
            scale=obj.scale,
            z=inferred_zdist.z,
            dndz=inferred_zdist.dndz,
            bias=None,
            mag_bias=None,
        )
        # pylint: enable=unexpected-keyword-arg

        return obj

    @final
    def _update_source(self, params: ParamsMap) -> None:
        """Perform any updates necessary after the parameters have being updated.

        This implementation must update all contained Updatable instances.

        :param params: the parameters to be used for the update
        """
        self.systematics.update(params)

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Return the derived parameters for this source.

        :return: the derived parameters
        """
        if self.derived_scale:
            assert self.current_tracer_args is not None
            derived_scale = DerivedParameter(
                "TwoPoint",
                f"NumberCountsScale_{self.sacc_tracer}",
                self.current_tracer_args.scale,
            )
            derived_parameters = DerivedParameterCollection([derived_scale])
        else:
            derived_parameters = DerivedParameterCollection([])

        return derived_parameters

    def _read(self, sacc_data) -> None:
        """Read the data for this source from the SACC file.

        :param sacc_data: The data in the sacc format.
        """
        # pylint: disable=unexpected-keyword-arg
        self.tracer_args = NumberCountsArgs(
            scale=self.scale,
            z=np.array([]),
            dndz=np.array([]),
            bias=None,
            mag_bias=None,
        )
        # pylint: enable=unexpected-keyword-arg
        super()._read(sacc_data)

    def create_tracers(
        self, tools: ModelingTools
    ) -> tuple[list[Tracer], NumberCountsArgs]:
        """Create the tracers for this source.

        :param tools: the ModelingTools used to create the tracers
        :return: a tuple of tracers and the updated tracer_args
        """
        tracer_args = self.tracer_args
        tracer_args = replace(tracer_args, bias=self.bias * np.ones_like(tracer_args.z))

        ccl_cosmo = tools.get_ccl_cosmology()
        for systematic in self.systematics:
            tracer_args = systematic.apply(tools, tracer_args)

        tracers = []

        if not tracer_args.has_pt or tracer_args.mag_bias is not None or self.has_rsd:
            # Create a normal pyccl.NumberCounts tracer if there's no PT, or
            # in case there's magnification or RSD.
            tracer_names = []
            if tracer_args.has_pt:
                # use PT for galaxy bias
                bias = None
            else:
                bias = (tracer_args.z, tracer_args.bias)
                tracer_names += ["galaxies"]
            if tracer_args.mag_bias is not None:
                tracer_names += ["magnification"]
            if self.has_rsd:
                tracer_names += ["rsd"]

            ccl_mag_tracer = pyccl.NumberCountsTracer(
                ccl_cosmo,
                has_rsd=self.has_rsd,
                dndz=(tracer_args.z, tracer_args.dndz),
                bias=bias,
                mag_bias=tracer_args.mag_bias,
            )

            tracers.append(
                Tracer(
                    ccl_mag_tracer,
                    tracer_name="+".join(tracer_names),
                    field=tracer_args.field,
                )
            )
        if tracer_args.has_pt:
            nc_pt_tracer = pyccl.nl_pt.PTNumberCountsTracer(
                b1=(tracer_args.z, tracer_args.bias),
                b2=tracer_args.b_2,
                bs=tracer_args.b_s,
            )

            ccl_nc_dummy_tracer = pyccl.NumberCountsTracer(
                ccl_cosmo,
                has_rsd=False,
                dndz=(tracer_args.z, tracer_args.dndz),
                bias=(tracer_args.z, np.ones_like(tracer_args.z)),
            )
            nc_pt_tracer = Tracer(
                ccl_nc_dummy_tracer, tracer_name="galaxies", pt_tracer=nc_pt_tracer
            )
            tracers.append(nc_pt_tracer)

        self.current_tracer_args = tracer_args

        return tracers, tracer_args

    def get_scale(self) -> float:
        """Return the scale for this source.

        :return: the scale for this source.
        """
        assert self.current_tracer_args
        return self.current_tracer_args.scale
