"""

Number counts source module
===========================

The classe in this file define ...

"""

from __future__ import annotations
from typing import List, Optional, final
from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

from .source import Source
from .source import Systematic
from .....parameters import ParamsMap, RequiredParameters, parameter_get_full_name
from .....updatable import UpdatableCollection

__all__ = ["NumberCounts"]


@dataclass(frozen=True)
class NumberCountsArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: np.ndarray
    dndz: np.ndarray
    bias: np.ndarray
    mag_bias: np.ndarray


class NumberCountsSystematic(Systematic):
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

    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["alphaz", "alphag", "z_piv"]
    alphaz: float
    alphag: float
    z_piv: float

    def __init__(self, sacc_tracer: str):
        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of parameters."""
        self.alphaz = params.get_from_prefix_param(self.sacc_tracer, "alphaz")
        self.alphag = params.get_from_prefix_param(self.sacc_tracer, "alphag")
        self.z_piv = params.get_from_prefix_param(self.sacc_tracer, "z_piv")

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

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

        return NumberCountsArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            bias=tracer_arg.bias * pref,
            mag_bias=tracer_arg.mag_bias,
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
        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of parameters."""
        self.r_lim = params.get_from_prefix_param(self.sacc_tracer, "r_lim")
        self.sig_c = params.get_from_prefix_param(self.sacc_tracer, "sig_c")
        self.eta = params.get_from_prefix_param(self.sacc_tracer, "eta")
        self.z_c = params.get_from_prefix_param(self.sacc_tracer, "z_c")
        self.z_m = params.get_from_prefix_param(self.sacc_tracer, "z_m")

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: NumberCountsArgs
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
        z = tracer_arg.z
        # The slope of log(n_tot(z,r_lim)) with respect to r_lim
        # where n_tot(z,r_lim) is the luminosity function after using fit (C.1)
        s = (
            self.eta / self.r_lim
            - 3.0 * self.z_m / z_bar
            + 1.5 * self.z_m * np.power(z / z_bar, 1.5) / z_bar
        )

        return NumberCountsArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            bias=tracer_arg.bias,
            mag_bias=tracer_arg.mag_bias * s / np.log(10),
        )


class PhotoZShift(NumberCountsSystematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.
    """

    params_names = ["delta_z"]
    delta_z: float

    def __init__(self, sacc_tracer: str):
        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.delta_z = params.get_from_prefix_param(self.sacc_tracer, "delta_z")

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: NumberCountsArgs):
        """Apply a shift to the photo-z distribution of a source."""

        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)

        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0

        return NumberCountsArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=dndz,
            bias=tracer_arg.bias,
            mag_bias=tracer_arg.mag_bias,
        )


class NumberCounts(Source):
    params_names = ["bias", "mag_bias"]
    bias: float
    mag_bias: Optional[float]

    systematics: UpdatableCollection
    tracer_arg: NumberCountsArgs

    def __init__(
        self,
        *,
        sacc_tracer,
        has_rsd=False,
        has_mag_bias=False,
        scale=1.0,
        systematics: Optional[List[NumberCountsSystematic]] = None,
    ):
        self.sacc_tracer = sacc_tracer
        self.has_rsd = has_rsd
        self.has_mag_bias = has_mag_bias

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

        if self.has_mag_bias:
            self.mag_bias = params.get_from_prefix_param(self.sacc_tracer, "mag_bias")
        else:
            self.mag_bias = None

        self.systematics.update(params)

    @final
    def required_parameters(self) -> RequiredParameters:
        if self.has_mag_bias:
            rp = RequiredParameters(
                [
                    parameter_get_full_name(self.sacc_tracer, pn)
                    for pn in self.params_names
                ]
            )
        else:
            rp = RequiredParameters(
                [
                    parameter_get_full_name(self.sacc_tracer, pn)
                    for pn in self.params_names
                    if pn != "mag_bias"
                ]
            )
        return rp + self.systematics.required_parameters()

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

    def create_tracer(self, cosmo: pyccl.Cosmology):
        tracer_args = self.tracer_args

        bias = np.ones_like(tracer_args.z) * self.bias
        tracer_args = NumberCountsArgs(
            scale=tracer_args.scale,
            z=tracer_args.z,
            dndz=tracer_args.dndz,
            bias=bias,
            mag_bias=tracer_args.mag_bias,
        )

        if self.mag_bias is not None:
            mag_bias = np.ones_like(tracer_args.z) * self.mag_bias
            tracer_args = NumberCountsArgs(
                scale=tracer_args.scale,
                z=tracer_args.z,
                dndz=tracer_args.dndz,
                bias=tracer_args.bias,
                mag_bias=mag_bias,
            )

        for systematic in self.systematics:
            tracer_args = systematic.apply(cosmo, tracer_args)

        if self.has_mag_bias:
            tracer = pyccl.NumberCountsTracer(
                cosmo,
                has_rsd=self.has_rsd,
                dndz=(tracer_args.z, tracer_args.dndz),
                bias=(tracer_args.z, tracer_args.bias),
                mag_bias=(tracer_args.z, tracer_args.mag_bias),
            )
        else:
            tracer = pyccl.NumberCountsTracer(
                cosmo,
                has_rsd=self.has_rsd,
                dndz=(tracer_args.z, tracer_args.dndz),
                bias=(tracer_args.z, tracer_args.bias),
            )
        self.current_tracer_args = tracer_args

        return tracer, tracer_args

    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale
