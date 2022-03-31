from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

from .source import Source
from .source import Systematic
from firecrown.parameters import get_from_prefix_param

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
    pass

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

    def __init__(self, alphaz, alphag, z_piv):
        self.alphaz = alphaz
        self.alphag = alphag
        self.z_piv = z_piv

    def apply(self, cosmo, params, source):
        """Apply a linear bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        pref = ((1.0 + source.z_) / (1.0 + params[self.z_piv])) ** params[self.alphaz]
        pref *= ccl.growth_factor(cosmo, 1.0 / (1.0 + source.z_)) ** params[self.alphag]
        source.bias_ *= pref


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

    def __init__(self, r_lim, sig_c, eta, z_c, z_m):
        self.r_lim = r_lim
        self.sig_c = sig_c
        self.eta = eta
        self.z_c = z_c
        self.z_m = z_m

    def apply(self, cosmo, params, source):
        """Apply a magnification bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """

        z_bar = params[self.z_c] + params[self.z_m] * (params[self.r_lim] - 24)
        z = source.z_
        # The slope of log(n_tot(z,r_lim)) with respect to r_lim
        # where n_tot(z,r_lim) is the luminosity function after using fit (C.1)
        s = (
            params[self.eta] / params[self.r_lim]
            - 3 * params[self.z_m] / z_bar
            + 1.5 * params[self.z_m] * np.power(z / z_bar, 1.5) / z_bar
        )
        source.mag_bias_ *= s / np.log(10)

class PhotoZShift(NumberCountsSystematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.
    """

    params_names = ["delta_z"]

    def __init__(self, sacc_tracer: str):
        self.sacc_tracer = sacc_tracer
        self.delta_z = None

    def update_params(self, params):
        self.delta_z = get_from_prefix_param(self, params, self.sacc_tracer, "delta_z")

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
        self.bias = None
        self.mag_bias = None
        self.has_rsd = has_rsd
        self.has_mag_bias = has_mag_bias

        self.systematics = []
        for systematic in systematics:
            self.systematics.append(systematic)

        self.scale = scale
        self.tracer_args = None
        self.current_tracer_args = None
        self.scale_ = None
        self.tracer_ = None

    def _update_params(self, params):
        self.bias = get_from_prefix_param(self, params, self.sacc_tracer, "bias")

        if self.has_mag_bias:
            self.mag_bias = get_from_prefix_param(
                self, params, self.sacc_tracer, "mag_bias"
            )

        for systematic in self.systematics:
            systematic.update_params(params)

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

    def create_tracer(self, cosmo: pyccl.Cosmology, params):
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
            tracer = ccl.NumberCountsTracer(
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
