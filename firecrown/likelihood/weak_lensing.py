"""Weak lensing source and systematics."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Sequence, final, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.nl_pt
import sacc


from firecrown import parameters
from firecrown.likelihood.source import (
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxyPhotoZShift,
    SourceGalaxyPhotoZShiftandStretch,
    SourceGalaxySelectField,
    SourceGalaxySystematic,
    PhotoZShiftFactory,
    PhotoZShiftandStretchFactory,
    Tracer,
)
from firecrown.metadata_types import InferredGalaxyZDist, TypeSource
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap


@dataclass(frozen=True)
class WeakLensingArgs(SourceGalaxyArgs):
    """Class for weak lensing tracer builder argument."""

    ia_bias: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None

    has_pt: bool = False
    has_hm: bool = False

    ia_pt_c_1: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    ia_pt_c_d: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    ia_pt_c_2: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None

    ia_a_1h: None | npt.NDArray[np.float64] = None
    ia_a_2h: None | npt.NDArray[np.float64] = None


class WeakLensingSystematic(SourceGalaxySystematic[WeakLensingArgs]):
    """Abstract base class for all weak lensing systematics."""

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply method to include systematics in the tracer_arg."""


class PhotoZShiftandStretch(SourceGalaxyPhotoZShiftandStretch[WeakLensingArgs]):
    """Photo-z shift systematic."""


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

    def __init__(self, sacc_tracer: None | str = None, alphag: None | float = 1.0):
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
TATT_ALIGNMENT_DEFAULT_IA_ZPIV_1 = 0.62
TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_1 = 0.0
TATT_ALIGNMENT_DEFAULT_IA_A_2 = 0.5
TATT_ALIGNMENT_DEFAULT_IA_ZPIV_2 = 0.62
TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_2 = 0.0
TATT_ALIGNMENT_DEFAULT_IA_A_D = 0.5
TATT_ALIGNMENT_DEFAULT_IA_ZPIV_D = 0.62
TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_D = 0.0


class TattAlignmentSystematic(WeakLensingSystematic):
    """TATT alignment systematic.

    This systematic adds a TATT (nonlinear) intrinsic alignment model systematic.

    The amplitude of each contribution to the TATT model
    (i.e. linear, density-dependent, or quadratic terms) can be expressed as
    a function in redshift, parameterized by the relationship:
    $A_i \times \frac{1 + z}{1 + z_{piv,i}}^{\alpha_i}$

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar ia_a_1: the amplitude of the linear alignment model.
    :ivar ia_zpiv_1: the pivot redshift of the linear alignment model.
    :ivar ia_alphaz_1: the redshift dependence of the linear alignment model.
    :ivar ia_a_2: the amplitude of the quadratic alignment model.
    :ivar ia_zpiv_2: the pivot redshift of the quadratic alignment model.
    :ivar ia_alphaz_2: the redshift dependence of the quadratic alignment model.
    :ivar ia_a_d: the amplitude of the density-dependent alignment model.
    :ivar ia_zpiv_d: the pivot redshift of the density-dependent alignment model.
    :ivar ia_alphaz_d: the redshift dependence of the density-dependent alignment model.
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
        self.ia_zpiv_1 = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ZPIV_1
        )
        self.ia_alphaz_1 = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_1
        )
        self.ia_a_2 = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_A_2
        )
        self.ia_zpiv_2 = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ZPIV_2
        )
        self.ia_alphaz_2 = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_2
        )
        self.ia_a_d = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_A_D
        )
        self.ia_zpiv_d = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ZPIV_D
        )
        self.ia_alphaz_d = parameters.register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_D
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

        c_1 *= ((1.0 + z) / (1.0 + self.ia_zpiv_1)) ** self.ia_alphaz_1
        c_d *= ((1.0 + z) / (1.0 + self.ia_zpiv_d)) ** self.ia_alphaz_d
        c_2 *= ((1.0 + z) / (1.0 + self.ia_zpiv_2)) ** self.ia_alphaz_2

        return replace(
            tracer_arg,
            has_pt=True,
            ia_pt_c_1=(z, c_1),
            ia_pt_c_d=(z, c_d),
            ia_pt_c_2=(z, c_2),
        )


HM_ALIGNMENT_DEFAULT_IA_A_1H = 1e-4
HM_ALIGNMENT_DEFAULT_IA_A_2H = 1.0


class HMAlignmentSystematic(WeakLensingSystematic):
    """Halo model intrinsic alignment systematic.

    This systematic adds a halo model based intrinsic alignment systematic
    which, at the moment, is fixed within the redshift bin.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar ia_a_1h: the 1-halo intrinsic alignment bias parameter (satellite galaxies).
    :ivar ia_a_2h: the 2-halo intrinsic alignment bias parameter (central galaxies).
    """

    def __init__(self, _: None | str = None):
        """Create a HMAlignmentSystematic object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__()

        self.ia_a_1h = parameters.register_new_updatable_parameter(
            default_value=HM_ALIGNMENT_DEFAULT_IA_A_1H
        )
        self.ia_a_2h = parameters.register_new_updatable_parameter(
            default_value=HM_ALIGNMENT_DEFAULT_IA_A_2H
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new halo-model alignment systematic.

        :param tools: A ModelingTools object.
        :param tracer_arg: The WeakLensingArgs to which apply the systematic.
        :returns: A new WeakLensingArgs object with the systematic applied.
        """
        return replace(
            tracer_arg, has_hm=True, ia_a_1h=self.ia_a_1h, ia_a_2h=self.ia_a_2h
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
        # pylint: disable=unexpected-keyword-arg
        obj.tracer_args = WeakLensingArgs(
            scale=obj.scale, z=inferred_zdist.z, dndz=inferred_zdist.dndz, ia_bias=None
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


class MultiplicativeShearBiasFactory(BaseModel):
    """Factory class for MultiplicativeShearBias objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["MultiplicativeShearBiasFactory"],
        Field(description="The type of the systematic."),
    ] = "MultiplicativeShearBiasFactory"

    def create(self, bin_name: str) -> MultiplicativeShearBias:
        """Create a MultiplicativeShearBias object.

        :param inferred_zdist: The inferred galaxy redshift distribution for
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

        :param inferred_zdist: The inferred galaxy redshift distribution for
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

    def create(self, bin_name: str) -> TattAlignmentSystematic:
        """Create a TattAlignmentSystematic object.

        :param inferred_zdist: The inferred galaxy redshift distribution for
            the created TattAlignmentSystematic object.
        :return: The created TattAlignmentSystematic object.
        """
        return TattAlignmentSystematic(bin_name)

    def create_global(self) -> TattAlignmentSystematic:
        """Create a TattAlignmentSystematic object.

        :return: The created TattAlignmentSystematic object.
        """
        return TattAlignmentSystematic(None)


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

    def create(self, inferred_zdist: InferredGalaxyZDist) -> WeakLensing:
        """Create a WeakLensing object with the given tracer name and scale."""
        inferred_zdist_id = id(inferred_zdist)
        if inferred_zdist_id in self._cache:
            return self._cache[inferred_zdist_id]

        systematics: list[SourceGalaxySystematic[WeakLensingArgs]] = [
            systematic_factory.create(inferred_zdist.bin_name)
            for systematic_factory in self.per_bin_systematics
        ]
        systematics.extend(self._global_systematics_instances)

        wl = WeakLensing.create_ready(inferred_zdist, systematics)
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
