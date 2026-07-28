"""Weak lensing source class."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np
import pyccl
import pyccl.nl_pt
import sacc

from firecrown.likelihood.weak_lensing._args import WeakLensingArgs
from firecrown.likelihood_base import (
    SourceGalaxy,
    SourceGalaxySystematic,
    Tracer,
)
from firecrown.metadata_types import TomographicBin
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import ParamsMap


class WeakLensing(SourceGalaxy[WeakLensingArgs]):
    """Source class for weak lensing."""

    def __init__(
        self,
        *,
        sacc_tracer: str,
        scale: float = 1.0,
        systematics: None | Sequence[SourceGalaxySystematic[WeakLensingArgs]] = None,
    ):
        """Initialize the WeakLensing object.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        :param scale: the scale of the source. This is used to scale the shear
            power spectrum.
        :param systematics: a list of WeakLensingSystematic objects to apply to
            this source.

        """
        super().__init__(sacc_tracer=sacc_tracer, systematics=systematics)

        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.current_tracer_args: None | WeakLensingArgs = None
        self.tracer_args: WeakLensingArgs

    @classmethod
    def create_ready(
        cls,
        tomographic_bin: TomographicBin,
        systematics: None | list[SourceGalaxySystematic[WeakLensingArgs]] = None,
    ) -> WeakLensing:
        """Create a WeakLensing object with the given tracer name and scale."""
        obj = cls(sacc_tracer=tomographic_bin.bin_name, systematics=systematics)
        # pylint: disable=unexpected-keyword-arg
        obj.tracer_args = WeakLensingArgs(
            scale=obj.scale,
            z=tomographic_bin.z,
            dndz=tomographic_bin.dndz,
            ia_bias=None,
        )
        # pylint: enable=unexpected-keyword-arg

        return obj

    @final
    def _update_source(self, params: ParamsMap):
        """Implementation of Source interface `_update_source`.

        This updates all the contained systematics.
        """
        self.systematics.update(params)

    def _read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this source from the SACC file.

        This sets self.tracer_args, based on the data in `sacc_data` associated with
        this object's `sacc_tracer` name.
        """
        # pylint: disable=unexpected-keyword-arg
        self.tracer_args = WeakLensingArgs(
            scale=self.scale, z=np.array([]), dndz=np.array([]), ia_bias=None
        )
        # pylint: enable=unexpected-keyword-arg

        super()._read(sacc_data)

    def create_tracers(self, tools: ModelingTools):
        """Render a source by applying systematics."""
        ccl_cosmo = tools.get_ccl_cosmology()
        tracer_args = self.tracer_args

        assert self.systematics is not None
        for systematic in self.systematics:
            tracer_args = systematic.apply(tools, tracer_args)

        ccl_wl_tracer = pyccl.WeakLensingTracer(
            ccl_cosmo,
            dndz=(tracer_args.z, tracer_args.dndz),
            ia_bias=tracer_args.ia_bias,
        )
        tracers = [Tracer(ccl_wl_tracer, tracer_name="shear", field=tracer_args.field)]

        if tracer_args.has_pt:
            ia_pt_tracer = pyccl.nl_pt.PTIntrinsicAlignmentTracer(
                c1=tracer_args.ia_pt_c_1,
                cdelta=tracer_args.ia_pt_c_d,
                c2=tracer_args.ia_pt_c_2,
            )

            ccl_wl_dummy_tracer = pyccl.WeakLensingTracer(
                ccl_cosmo,
                has_shear=False,
                use_A_ia=False,
                dndz=(tracer_args.z, tracer_args.dndz),
                ia_bias=(tracer_args.z, np.ones_like(tracer_args.z)),
            )
            ia_tracer = Tracer(
                ccl_wl_dummy_tracer, tracer_name="intrinsic_pt", pt_tracer=ia_pt_tracer
            )
            tracers.append(ia_tracer)

        if tracer_args.has_hm:
            hmc = tools.get_hm_calculator()
            cM = tools.get_cM_relation()
            halo_profile = pyccl.halos.SatelliteShearHOD(
                mass_def=hmc.mass_def, concentration=cM, a1h=tracer_args.ia_a_1h
            )
            ccl_wl_dummy_tracer = pyccl.WeakLensingTracer(
                ccl_cosmo,
                has_shear=False,
                use_A_ia=False,
                dndz=(tracer_args.z, tracer_args.dndz),
                ia_bias=(tracer_args.z, np.ones_like(tracer_args.z)),
            )
            ia_tracer = Tracer(
                ccl_wl_dummy_tracer,
                tracer_name="intrinsic_alignment_hm",
                halo_profile=halo_profile,
            )
            # TODO: redesign this so that we are not adding a new
            # attribute to a pyccl class.
            halo_profile.ia_a_2h = (
                tracer_args.ia_a_2h
            )  # Attach the 2-halo amplitude here.
            tracers.append(ia_tracer)

        self.current_tracer_args = tracer_args

        return tracers, tracer_args

    def get_scale(self):
        """Returns the scales for this Source."""
        assert self.current_tracer_args
        return self.current_tracer_args.scale
