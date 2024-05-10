"""Weak lensing source and systematics."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Sequence, final

import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.nl_pt
import sacc

from firecrown import parameters
from firecrown.likelihood.gauss_family.statistic.source.source import (
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxyPhotoZShift,
    SourceGalaxySelectField,
    SourceGalaxySystematic,
    Tracer,
)
from firecrown.metadata.two_point import InferredGalaxyZDist
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap

__all__ = ["WeakLensing"]


@dataclass(frozen=True)
class WeakLensingArgs(SourceGalaxyArgs):
    """Class for weak lensing tracer builder argument."""

    ia_bias: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None

    has_pt: bool = False
    has_hm: bool = False

    ia_pt_c_1: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    ia_pt_c_d: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    ia_pt_c_2: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None


class WeakLensingSystematic(SourceGalaxySystematic[WeakLensingArgs]):
    """Abstract base class for all weak lensing systematics."""

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply method to include systematics in the tracer_arg."""


class PhotoZShift(SourceGalaxyPhotoZShift[WeakLensingArgs]):
    """Photo-z shift systematic."""


class SelectField(SourceGalaxySelectField[WeakLensingArgs]):
    """Systematic to select 3D field."""


MULTIPLICATIVE_SHEAR_BIAS_DEFAULT_BIAS = 1.0


class MultiplicativeShearBias(WeakLensingSystematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar mult_bias: the multiplicative shear bias parameter.
    """

    def __init__(self, sacc_tracer: str) -> None:
        """Create a MultiplicativeShearBias object that uses the named tracer.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.mult_bias = parameters.register_new_updatable_parameter(
            default_value=MULTIPLICATIVE_SHEAR_BIAS_DEFAULT_BIAS
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply multiplicative shear bias to a source.

        The `scale_` of the source is multiplied by `(1 + m)`.

        :param tools: A ModelingTools object.
        :param tracer_arg: The WeakLensingArgs to which apply the shear bias.

        :returns: A new WeakLensingArgs object with the shear bias applied.
        """
        return replace(
            tracer_arg,
            scale=tracer_arg.scale * (1.0 + self.mult_bias),
        )


LINEAR_ALIGNMENT_DEFAULT_IA_BIAS = 0.5
LINEAR_ALIGNMENT_DEFAULT_ALPHAZ = 0.0
LINEAR_ALIGNMENT_DEFAULT_ALPHAG = 1.0
LINEAR_ALIGNMENT_DEFAULT_Z_PIV = 0.5


class LinearAlignmentSystematic(WeakLensingSystematic):
    """Linear alignment systematic.

    This systematic adds a linear intrinsic alignment model systematic
    which varies with redshift and the growth function.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar ia_bias: the intrinsic alignment bias parameter.
    :ivar alphaz: the redshift dependence of the intrinsic alignment bias.
    :ivar alphag: the growth function dependence of the intrinsic alignment bias.
    :ivar z_piv: the pivot redshift for the intrinsic alignment bias.
    """

    def __init__(self, sacc_tracer: None | str = None, alphag=1.0):
        """Create a LinearAlignmentSystematic object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.ia_bias = parameters.register_new_updatable_parameter(
            default_value=LINEAR_ALIGNMENT_DEFAULT_IA_BIAS
        )
        self.alphaz = parameters.register_new_updatable_parameter(
            default_value=LINEAR_ALIGNMENT_DEFAULT_ALPHAZ
        )
        self.alphag = parameters.register_new_updatable_parameter(
            alphag, default_value=LINEAR_ALIGNMENT_DEFAULT_ALPHAG
        )
        self.z_piv = parameters.register_new_updatable_parameter(
            default_value=LINEAR_ALIGNMENT_DEFAULT_Z_PIV
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic.

        This choice is based on the given tracer_arg, in the context of the given
        cosmology.
        """
        ccl_cosmo = tools.get_ccl_cosmology()

        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= pyccl.growth_factor(ccl_cosmo, 1.0 / (1.0 + tracer_arg.z)) ** (
            self.alphag - 1.0
        )

        ia_bias_array = pref * self.ia_bias

        return replace(
            tracer_arg,
            ia_bias=(tracer_arg.z, ia_bias_array),
        )


TATT_ALIGNMENT_DEFAULT_IA_A_1 = 1.0
TATT_ALIGNMENT_DEFAULT_IA_A_2 = 0.5
TATT_ALIGNMENT_DEFAULT_IA_A_D = 0.5


class TattAlignmentSystematic(WeakLensingSystematic):
    """TATT alignment systematic.

    This systematic adds a TATT (nonlinear) intrinsic alignment model systematic.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar ia_a_1: the amplitude of the linear alignment model.
    :ivar ia_a_2: the amplitude of the quadratic alignment model.
    :ivar ia_a_d: the amplitude of the density-dependent alignment model.
    """

    def __init__(self, sacc_tracer: None | str = None):
        """Create a TattAlignmentSystematic object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)
        self.ia_a_1 = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_A_1
        )
        self.ia_a_2 = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_A_2
        )
        self.ia_a_d = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_A_D
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic.

        This choice is based on the given tracer_arg, in the context of the given
        cosmology.
        """
        ccl_cosmo = tools.get_ccl_cosmology()
        z = tracer_arg.z
        c_1, c_d, c_2 = pyccl.nl_pt.translate_IA_norm(
            ccl_cosmo,
            z=z,
            a1=self.ia_a_1,
            a1delta=self.ia_a_d,
            a2=self.ia_a_2,
            Om_m2_for_c2=False,
        )

        return replace(
            tracer_arg,
            has_pt=True,
            ia_pt_c_1=(z, c_1),
            ia_pt_c_d=(z, c_d),
            ia_pt_c_2=(z, c_2),
        )


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
        inferred_zdist: InferredGalaxyZDist,
        systematics: None | list[SourceGalaxySystematic[WeakLensingArgs]] = None,
    ) -> WeakLensing:
        """Create a WeakLensing object with the given tracer name and scale."""
        obj = cls(sacc_tracer=inferred_zdist.bin_name, systematics=systematics)
        obj.tracer_args = WeakLensingArgs(
            scale=obj.scale, z=inferred_zdist.z, dndz=inferred_zdist.dndz, ia_bias=None
        )

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
        self.tracer_args = WeakLensingArgs(
            scale=self.scale, z=np.array([]), dndz=np.array([]), ia_bias=None
        )

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

        self.current_tracer_args = tracer_args

        return tracers, tracer_args

    def get_scale(self):
        """Returns the scales for this Source."""
        assert self.current_tracer_args
        return self.current_tracer_args.scale


class WeakLensingSystematicFactory:
    """Factory class for WeakLensingSystematic objects."""

    @abstractmethod
    def create(
        self, inferred_zdist: InferredGalaxyZDist
    ) -> SourceGalaxySystematic[WeakLensingArgs]:
        """Create a WeakLensingSystematic object with the given tracer name."""


class MultiplicativeShearBiasFactory(WeakLensingSystematicFactory):
    """Factory class for MultiplicativeShearBias objects."""

    def create(self, inferred_zdist: InferredGalaxyZDist) -> MultiplicativeShearBias:
        return MultiplicativeShearBias(inferred_zdist.bin_name)


class TattAlignmentSystematicFactory(WeakLensingSystematicFactory):
    """Factory class for TattAlignmentSystematic objects."""

    def create(self, inferred_zdist: InferredGalaxyZDist) -> TattAlignmentSystematic:
        return TattAlignmentSystematic(inferred_zdist.bin_name)


class PhotoZShiftFactory(WeakLensingSystematicFactory):
    """Factory class for PhotoZShift objects."""

    def create(self, inferred_zdist: InferredGalaxyZDist) -> PhotoZShift:
        return PhotoZShift(inferred_zdist.bin_name)


class WeakLensingFactory:
    """Factory class for WeakLensing objects."""

    def __init__(
        self,
        per_bin_systematics: list[WeakLensingSystematicFactory],
        global_systematics: Sequence[WeakLensingSystematic],
    ) -> None:
        self.per_bin_systematics: list[WeakLensingSystematicFactory] = (
            per_bin_systematics
        )
        self.global_systematics: Sequence[WeakLensingSystematic] = global_systematics
        self.cache: dict[int, WeakLensing] = {}

    def create(self, inferred_zdist: InferredGalaxyZDist) -> WeakLensing:
        """Create a WeakLensing object with the given tracer name and scale."""
        inferred_zdist_id = id(inferred_zdist)
        if inferred_zdist_id in self.cache:
            return self.cache[inferred_zdist_id]

        systematics: list[SourceGalaxySystematic[WeakLensingArgs]] = [
            systematic_factory.create(inferred_zdist)
            for systematic_factory in self.per_bin_systematics
        ]
        systematics.extend(self.global_systematics)

        wl = WeakLensing.create_ready(inferred_zdist, systematics)
        self.cache[inferred_zdist_id] = wl

        return wl
