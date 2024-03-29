"""Number counts source and systematics."""

from __future__ import annotations
from typing import Optional, final
from dataclasses import dataclass, replace
from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import pyccl

from .source import (
    Tracer,
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxySystematic,
    SourceGalaxyPhotoZShift,
    SourceGalaxySelectField,
)

from ..... import parameters

from .....modeling_tools import ModelingTools
from .....parameters import (
    ParamsMap,
    DerivedParameter,
    DerivedParameterCollection,
)
from .....updatable import UpdatableCollection

__all__ = ["NumberCounts"]


@dataclass(frozen=True)
class NumberCountsArgs(SourceGalaxyArgs):
    """Class for number counts tracer builder argument."""

    bias: Optional[npt.NDArray[np.float64]] = None
    mag_bias: Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    has_pt: bool = False
    has_hm: bool = False
    b_2: Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    b_s: Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None


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

        self.alphaz = parameters.register_new_updatable_parameter()
        self.alphag = parameters.register_new_updatable_parameter()
        self.z_piv = parameters.register_new_updatable_parameter()

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

    def __init__(self, sacc_tracer: str):
        """Initialize the PTNonLinearBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.b_2 = parameters.register_new_updatable_parameter()
        self.b_s = parameters.register_new_updatable_parameter()

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
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

        self.r_lim = parameters.register_new_updatable_parameter()
        self.sig_c = parameters.register_new_updatable_parameter()
        self.eta = parameters.register_new_updatable_parameter()
        self.z_c = parameters.register_new_updatable_parameter()
        self.z_m = parameters.register_new_updatable_parameter()

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

        self.mag_bias = parameters.register_new_updatable_parameter()

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        return replace(
            tracer_arg,
            mag_bias=(tracer_arg.z, np.ones_like(tracer_arg.z) * self.mag_bias),
        )


class NumberCounts(SourceGalaxy[NumberCountsArgs]):
    """Source class for number counts."""

    def __init__(
        self,
        *,
        sacc_tracer: str,
        has_rsd: bool = False,
        derived_scale: bool = False,
        scale: float = 1.0,
        systematics: Optional[list[SourceGalaxySystematic[NumberCountsArgs]]] = None,
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

        self.bias = parameters.register_new_updatable_parameter()
        self.systematics: UpdatableCollection[
            SourceGalaxySystematic[NumberCountsArgs]
        ] = UpdatableCollection(systematics)
        self.scale = scale
        self.current_tracer_args: Optional[NumberCountsArgs] = None
        self.tracer_args: NumberCountsArgs

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
        self.tracer_args = NumberCountsArgs(
            scale=self.scale,
            z=np.array([]),
            dndz=np.array([]),
            bias=None,
            mag_bias=None,
        )
        super()._read(sacc_data)

    def create_tracers(self, tools: ModelingTools):
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
