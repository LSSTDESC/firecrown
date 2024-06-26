"""Number counts source and systematics."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Sequence, final, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

import numpy as np
import numpy.typing as npt
import pyccl

# firecrown is needed for backward compatibility; remove support for deprecated
# directory structure is removed.
import firecrown  # pylint: disable=unused-import # noqa: F401
from firecrown import parameters
from firecrown.likelihood.source import (
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxyPhotoZShift,
    SourceGalaxySelectField,
    SourceGalaxySystematic,
    Tracer,
)
from firecrown.metadata.two_point import InferredGalaxyZDist
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import DerivedParameter, DerivedParameterCollection, ParamsMap
from firecrown.updatable import UpdatableCollection


@dataclass(frozen=True)
class NumberCountsArgs(SourceGalaxyArgs):
    """Class for number counts tracer builder argument."""

    bias: None | npt.NDArray[np.float64] = None
    mag_bias: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    has_pt: bool = False
    has_hm: bool = False
    b_2: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None
    b_s: None | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None


class NumberCountsSystematic(SourceGalaxySystematic[NumberCountsArgs]):
    """Abstract base class for systematics for Number Counts sources.

    Derived classes must implement :python`apply` with the correct signature.
    """

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply method to include systematics in the tracer_arg."""


class PhotoZShift(SourceGalaxyPhotoZShift[NumberCountsArgs]):
    """Photo-z shift systematic."""


class SelectField(SourceGalaxySelectField[NumberCountsArgs]):
    """Systematic to select 3D field."""


LINEAR_BIAS_DEFAULT_ALPHAZ = 0.0
LINEAR_BIAS_DEFAULT_ALPHAG = 1.0
LINEAR_BIAS_DEFAULT_Z_PIV = 0.5


class LinearBiasSystematic(NumberCountsSystematic):
    """Linear bias systematic.

    This systematic adds a linear bias model which varies with redshift and
    the growth function.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar alphaz: the redshift exponent of the bias.
    :ivar alphag: the growth function exponent of the bias.
    :ivar z_piv: the pivot redshift of the bias.
    """

    def __init__(self, sacc_tracer: str):
        """Initialize the LinearBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.alphaz = parameters.register_new_updatable_parameter(
            default_value=LINEAR_BIAS_DEFAULT_ALPHAZ
        )
        self.alphag = parameters.register_new_updatable_parameter(
            default_value=LINEAR_BIAS_DEFAULT_ALPHAG
        )
        self.z_piv = parameters.register_new_updatable_parameter(
            default_value=LINEAR_BIAS_DEFAULT_Z_PIV
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a linear bias systematic.

        Parameters
        ----------
        cosmo : Cosmology
            A Cosmology object.
        tracer_arg : NumberCountsArgs
            The source to which apply the shear bias.
        """
        ccl_cosmo = tools.get_ccl_cosmology()
        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= (
            pyccl.growth_factor(ccl_cosmo, 1.0 / (1.0 + tracer_arg.z)) ** self.alphag
        )

        if tracer_arg.bias is None:
            bias = np.ones_like(tracer_arg.z)
        else:
            bias = tracer_arg.bias
        bias = bias * pref

        return replace(
            tracer_arg,
            bias=bias,
        )


PT_NON_LINEAR_BIAS_DEFAULT_B_2 = 1.0
PT_NON_LINEAR_BIAS_DEFAULT_B_S = 1.0


class PTNonLinearBiasSystematic(NumberCountsSystematic):
    """Non-linear bias systematic.

    This systematic adds a linear bias model which varies with redshift and
    the growth function.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar b_2: the quadratic bias.
    :ivar b_s: the stochastic bias.
    """

    def __init__(self, sacc_tracer: None | str = None):
        """Initialize the PTNonLinearBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.b_2 = parameters.register_new_updatable_parameter(
            default_value=PT_NON_LINEAR_BIAS_DEFAULT_B_2
        )
        self.b_s = parameters.register_new_updatable_parameter(
            default_value=PT_NON_LINEAR_BIAS_DEFAULT_B_S
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a non-linear bias systematic.

        :param tools: currently unused, but required by interface
        :param tracer_arg: a NumberCountsArgs object with values to be updated

        :return: the updated NumberCountsArgs object
        """
        z = tracer_arg.z
        b_2_z = self.b_2 * np.ones_like(z)
        b_s_z = self.b_s * np.ones_like(z)
        # b_1 uses the "bias" field
        return replace(
            tracer_arg,
            has_pt=True,
            b_2=(z, b_2_z),
            b_s=(z, b_s_z),
        )


class MagnificationBiasSystematic(NumberCountsSystematic):
    """Magnification bias systematic.

    This systematic adds a magnification bias model for galaxy number contrast
    following Joachimi & Bridle (2010), arXiv:0911.2454.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar r_lim: the limiting magnitude.
    :ivar sig_c: the intrinsic dispersion of the source redshift distribution.
    :ivar eta: the slope of the luminosity function.
    :ivar z_c: the characteristic redshift of the source distribution.
    :ivar z_m: the slope of the source redshift distribution.
    """

    def __init__(self, sacc_tracer: str):
        """Initialize the MagnificationBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.r_lim = parameters.register_new_updatable_parameter(default_value=24.0)
        self.sig_c = parameters.register_new_updatable_parameter(default_value=9.83)
        self.eta = parameters.register_new_updatable_parameter(default_value=19.0)
        self.z_c = parameters.register_new_updatable_parameter(default_value=0.39)
        self.z_m = parameters.register_new_updatable_parameter(default_value=0.055)

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a magnification bias systematic.

        :param tools: a ModelingTools object
        :param tracer_arg: a NumberCountsArgs object

        :return: a NumberCountsArgs object
        """
        z_bar = self.z_c + self.z_m * (self.r_lim - 24.0)
        # The slope of log(n_tot(z,r_lim)) with respect to r_lim
        # where n_tot(z,r_lim) is the luminosity function after using fit (C.1)
        s = (
            self.eta / self.r_lim
            - 3.0 * self.z_m / z_bar
            + 1.5 * self.z_m * np.power(tracer_arg.z / z_bar, 1.5) / z_bar
        )

        if tracer_arg.mag_bias is None:
            mag_bias = np.ones_like(tracer_arg.z)
        else:
            mag_bias = tracer_arg.mag_bias[1]
        mag_bias = mag_bias * s / np.log(10)

        return replace(
            tracer_arg,
            mag_bias=(tracer_arg.z, mag_bias),
        )


CONSTANT_MAGNIFICATION_BIAS_DEFAULT_MAG_BIAS = 1.0


class ConstantMagnificationBiasSystematic(NumberCountsSystematic):
    """Simple constant magnification bias systematic.

    This systematic adds a constant magnification bias model for galaxy number
    contrast.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar mag_bias: the magnification bias.
    """

    def __init__(self, sacc_tracer: str):
        """Initialize the ConstantMagnificationBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.mag_bias = parameters.register_new_updatable_parameter(
            default_value=CONSTANT_MAGNIFICATION_BIAS_DEFAULT_MAG_BIAS
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a constant magnification bias systematic.

        :param tools: currently unused, but required by interface
        :param tracer_arg: a NumberCountsArgs object with values to be updated

        :return: the updated NumberCountsArgs object
        """
        return replace(
            tracer_arg,
            mag_bias=(tracer_arg.z, np.ones_like(tracer_arg.z) * self.mag_bias),
        )


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
        """Create a NumberCounts object with the given tracer name and scale."""
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
    def _update_source(self, params: ParamsMap):
        """Perform any updates necessary after the parameters have being updated.

        This implementation must update all contained Updatable instances.
        """
        self.systematics.update(params)

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
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

    def _read(self, sacc_data):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
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
        """Create the tracers for this source."""
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

    def get_scale(self):
        """Return the scale for this source."""
        assert self.current_tracer_args
        return self.current_tracer_args.scale


class PhotoZShiftFactory(BaseModel):
    """Factory class for PhotoZShift objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["PhotoZShiftFactory"],
        Field(description="The type of the systematic."),
    ] = "PhotoZShiftFactory"

    def create(self, inferred_zdist: InferredGalaxyZDist) -> PhotoZShift:
        """Create a PhotoZShift object with the given tracer name."""
        return PhotoZShift(inferred_zdist.bin_name)

    def create_global(self) -> PhotoZShift:
        """Create a PhotoZShift object with the given tracer name."""
        raise ValueError("PhotoZShift cannot be global.")


class LinearBiasSystematicFactory(BaseModel):
    """Factory class for LinearBiasSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["LinearBiasSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "LinearBiasSystematicFactory"

    def create(self, inferred_zdist: InferredGalaxyZDist) -> LinearBiasSystematic:
        """Create a LinearBiasSystematic object with the given tracer name."""
        return LinearBiasSystematic(inferred_zdist.bin_name)

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

    def create(self, inferred_zdist: InferredGalaxyZDist) -> PTNonLinearBiasSystematic:
        """Create a PTNonLinearBiasSystematic object with the given tracer name."""
        return PTNonLinearBiasSystematic(inferred_zdist.bin_name)

    def create_global(self) -> PTNonLinearBiasSystematic:
        """Create a PTNonLinearBiasSystematic object with the given tracer name."""
        return PTNonLinearBiasSystematic()


class MagnificationBiasSystematicFactory(BaseModel):
    """Factory class for MagnificationBiasSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["MagnificationBiasSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "MagnificationBiasSystematicFactory"

    def create(
        self, inferred_zdist: InferredGalaxyZDist
    ) -> MagnificationBiasSystematic:
        """Create a MagnificationBiasSystematic object with the given tracer name."""
        return MagnificationBiasSystematic(inferred_zdist.bin_name)

    def create_global(self) -> MagnificationBiasSystematic:
        """Create a MagnificationBiasSystematic object with the given tracer name."""
        raise ValueError("MagnificationBiasSystematic cannot be global.")


class ConstantMagnificationBiasSystematicFactory(BaseModel):
    """Factory class for ConstantMagnificationBiasSystematic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["ConstantMagnificationBiasSystematicFactory"],
        Field(description="The type of the systematic."),
    ] = "ConstantMagnificationBiasSystematicFactory"

    def create(
        self, inferred_zdist: InferredGalaxyZDist
    ) -> ConstantMagnificationBiasSystematic:
        """Create a ConstantMagnificationBiasSystematic object.

        Use the inferred_zdist to create the systematic.
        """
        return ConstantMagnificationBiasSystematic(inferred_zdist.bin_name)

    def create_global(self) -> ConstantMagnificationBiasSystematic:
        """Create a ConstantMagnificationBiasSystematic object.

        Use the inferred_zdist to create the systematic.
        """
        raise ValueError("ConstantMagnificationBiasSystematic cannot be global.")


NumberCountsSystematicFactory = Annotated[
    PhotoZShiftFactory
    | LinearBiasSystematicFactory
    | PTNonLinearBiasSystematicFactory
    | MagnificationBiasSystematicFactory
    | ConstantMagnificationBiasSystematicFactory,
    Field(discriminator="type", union_mode="left_to_right"),
]


class NumberCountsFactory(BaseModel):
    """Factory class for NumberCounts objects."""

    _cache: dict[int, NumberCounts] = PrivateAttr()
    _global_systematics_instances: Sequence[
        SourceGalaxySystematic[NumberCountsArgs]
    ] = PrivateAttr()

    per_bin_systematics: Sequence[NumberCountsSystematicFactory]
    global_systematics: Sequence[NumberCountsSystematicFactory]

    def model_post_init(self, __context) -> None:
        """Initialize the NumberCountsFactory."""
        self._cache: dict[int, NumberCounts] = {}
        self._global_systematics_instances = [
            nc_systematic_factory.create_global()
            for nc_systematic_factory in self.global_systematics
        ]

    def create(self, inferred_zdist: InferredGalaxyZDist) -> NumberCounts:
        """Create a NumberCounts object with the given tracer name and scale."""
        inferred_zdist_id = id(inferred_zdist)
        if inferred_zdist_id in self._cache:
            return self._cache[inferred_zdist_id]

        systematics: list[SourceGalaxySystematic[NumberCountsArgs]] = [
            systematic_factory.create(inferred_zdist)
            for systematic_factory in self.per_bin_systematics
        ]
        systematics.extend(self._global_systematics_instances)

        nc = NumberCounts.create_ready(inferred_zdist, systematics=systematics)
        self._cache[inferred_zdist_id] = nc

        return nc
