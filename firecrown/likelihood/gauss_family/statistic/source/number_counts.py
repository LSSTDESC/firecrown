"""Number counts source and systematics

"""

from __future__ import annotations
from typing import List, Tuple, Optional, final
from dataclasses import dataclass, replace
from abc import abstractmethod

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

from .....likelihood.likelihood import Cosmology

from .source import Source
from .source import SourceSystematic
from .source import Tracer

from .....parameters import (
    ParamsMap,
    RequiredParameters,
    parameter_get_full_name,
    DerivedParameterScalar,
    DerivedParameterCollection,
)
from .....updatable import UpdatableCollection

__all__ = ["NumberCounts"]


@dataclass(frozen=True)
class NumberCountsArgs:
    """Class for number counts tracer builder argument."""

    scale: float
    z: np.ndarray  # pylint: disable-msg=invalid-name
    dndz: np.ndarray
    bias: Optional[np.ndarray] = None
    mag_bias: Optional[Tuple[np.ndarray, np.ndarray]] = None
    has_pt: bool = False
    has_hm: bool = False
    b_2: Optional[Tuple[np.ndarray, np.ndarray]] = None
    b_s: Optional[Tuple[np.ndarray, np.ndarray]] = None


class NumberCountsSystematic(SourceSystematic):
    """Class implementing systematics for Number Counts sources."""

    @abstractmethod
    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply method to include systematics in the tracer_arg."""


class LinearBiasSystematic(NumberCountsSystematic):
    """Linear bias systematic.

    This systematic adds a linear bias model which varies with redshift and
    the growth function.

    Parameters
    ----------
    alphaz : str
        The mame of redshift dependence parameter of the linear bias.
    alphag : str
        The name of the growth dependence parameter of the linear bias.
    z_piv : str
        The name of the pivot redshift parameter for the linear bias.

    """

    params_names = ["alphaz", "alphag", "z_piv"]
    alphaz: float
    alphag: float
    z_piv: float

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of
        parameters."""
        self.alphaz = params.get_from_prefix_param(self.sacc_tracer, "alphaz")
        self.alphag = params.get_from_prefix_param(self.sacc_tracer, "alphag")
        self.z_piv = params.get_from_prefix_param(self.sacc_tracer, "z_piv")

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a linear bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        tracer_arg : NumberCountsArgs
            The source to which apply the shear bias.
        """

        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= pyccl.growth_factor(cosmo, 1.0 / (1.0 + tracer_arg.z)) ** self.alphag

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
    """

    params_names = ["b_2", "b_s"]
    b_2: float
    b_s: float

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of
        parameters."""
        self.b_2 = params.get_from_prefix_param(self.sacc_tracer, "b_2")
        self.b_s = params.get_from_prefix_param(self.sacc_tracer, "b_s")

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, cosmo: Cosmology, tracer_arg: NumberCountsArgs
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

    Parameters
    ----------
    r_lim : str
        The name of the limiting magnitude in r band filter.
    sig_c, eta, z_c, z_m : str
        The name of the fitting parameters in Joachimi & Bridle (2010) equation
        (C.1).

    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["r_lim", "sig_c", "eta", "z_c", "z_m"]
    r_lim: float
    sig_c: float
    eta: float
    z_c: float
    z_m: float

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of
        parameters."""
        self.r_lim = params.get_from_prefix_param(self.sacc_tracer, "r_lim")
        self.sig_c = params.get_from_prefix_param(self.sacc_tracer, "sig_c")
        self.eta = params.get_from_prefix_param(self.sacc_tracer, "eta")
        self.z_c = params.get_from_prefix_param(self.sacc_tracer, "z_c")
        self.z_m = params.get_from_prefix_param(self.sacc_tracer, "z_m")

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, cosmo: Cosmology, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a magnification bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        tracer_arg : NumberCountsArgs
            The source to which apply the shear bias.
        """

        z_bar = self.z_c + self.z_m * (self.r_lim - 24.0)
        # The slope of log(n_tot(z,r_lim)) with respect to r_lim
        # where n_tot(z,r_lim) is the luminosity function after using fit (C.1)
        # pylint: disable-next=invalid-name
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

    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["mag_bias"]
    mag_bias: float

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of
        parameters."""
        self.mag_bias = params.get_from_prefix_param(self.sacc_tracer, "mag_bias")

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, cosmo: Cosmology, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:

        return replace(
            tracer_arg,
            mag_bias=(tracer_arg.z, np.ones_like(tracer_arg.z) * self.mag_bias),
        )


class PhotoZShift(NumberCountsSystematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.
    """

    params_names = ["delta_z"]
    delta_z: float

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.delta_z = params.get_from_prefix_param(self.sacc_tracer, "delta_z")

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: NumberCountsArgs):
        """Apply a shift to the photo-z distribution of a source."""

        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)

        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0

        return replace(
            tracer_arg,
            dndz=dndz,
        )


class NumberCounts(Source):
    """Source class for number counts."""

    params_names = ["bias"]
    bias: float

    systematics: UpdatableCollection
    tracer_arg: NumberCountsArgs

    def __init__(
        self,
        *,
        sacc_tracer: str,
        has_rsd: bool = False,
        derived_scale: bool = False,
        scale: float = 1.0,
        systematics: Optional[List[NumberCountsSystematic]] = None,
    ):
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.has_rsd = has_rsd
        self.derived_scale = derived_scale

        self.systematics = UpdatableCollection([])
        if systematics:
            for systematic in systematics:
                self.systematics.append(systematic)

        self.scale = scale
        self.current_tracer_args = None
        self.scale_ = None
        self.tracer_ = None

    @final
    def _update_source(self, params: ParamsMap):
        self.bias = params.get_from_prefix_param(self.sacc_tracer, "bias")
        self.systematics.update(params)

    @final
    def _reset_source(self) -> None:
        self.systematics.reset()

    @final
    def required_parameters(self) -> RequiredParameters:
        rp = RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )
        return rp + self.systematics.required_parameters()

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        if self.derived_scale:
            assert self.current_tracer_args is not None
            derived_scale = DerivedParameterScalar(
                "TwoPoint",
                f"NumberCountsScale_{self.sacc_tracer}",
                self.current_tracer_args.scale,
            )
            derived_parameters = DerivedParameterCollection([derived_scale])
        else:
            derived_parameters = DerivedParameterCollection([])
        derived_parameters = (
            derived_parameters + self.systematics.get_derived_parameters()
        )

        return derived_parameters

    def _read(self, sacc_data):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        tracer = sacc_data.get_tracer(self.sacc_tracer)
        z = getattr(tracer, "z").copy().flatten()
        nz = getattr(tracer, "nz").copy().flatten()
        inds = np.argsort(z)
        z = z[inds]
        nz = nz[inds]

        self.tracer_args = NumberCountsArgs(
            scale=self.scale, z=z, dndz=nz, bias=None, mag_bias=None
        )

    def create_tracers(self, cosmo: Cosmology):
        tracer_args = self.tracer_args
        tracer_args = replace(tracer_args, bias=self.bias * np.ones_like(tracer_args.z))

        for systematic in self.systematics:
            tracer_args = systematic.apply(cosmo, tracer_args)

        tracers = []

        if tracer_args.has_pt:
            nc_pt_tracer = pyccl.nl_pt.PTNumberCountsTracer(
                b1=(tracer_args.z, tracer_args.bias),
                b2=tracer_args.b_2,
                bs=tracer_args.b_s,
            )

            ccl_nc_dummy_tracer = pyccl.NumberCountsTracer(
                cosmo.ccl_cosmo,
                has_rsd=False,
                dndz=(tracer_args.z, tracer_args.dndz),
                bias=(tracer_args.z, np.ones_like(tracer_args.z)),
            )
            nc_pt_tracer = Tracer(
                ccl_nc_dummy_tracer, field="galaxies", pt_tracer=nc_pt_tracer
            )
            tracers.append(nc_pt_tracer)

            if tracer_args.mag_bias is not None or self.has_rsd:
                matter_pt_tracer = pyccl.nl_pt.PTMatterTracer()
                ccl_mag_tracer = pyccl.NumberCountsTracer(
                    cosmo.ccl_cosmo,
                    has_rsd=self.has_rsd,
                    dndz=(tracer_args.z, tracer_args.dndz),
                    bias=None,
                    mag_bias=tracer_args.mag_bias,
                )
                field_name = "magnification"
                if self.has_rsd:
                    field_name += "+rsd"
                mag_pt_tracer = Tracer(
                    ccl_mag_tracer, field=field_name, pt_tracer=matter_pt_tracer
                )
                tracers.append(mag_pt_tracer)
        else:
            ccl_nc_tracer = pyccl.NumberCountsTracer(
                cosmo.ccl_cosmo,
                has_rsd=self.has_rsd,
                dndz=(tracer_args.z, tracer_args.dndz),
                bias=(tracer_args.z, tracer_args.bias),
                mag_bias=tracer_args.mag_bias,
            )
            tracers.append(Tracer(ccl_nc_tracer, field="galaxies"))

        self.current_tracer_args = tracer_args

        return tracers, tracer_args

    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale
