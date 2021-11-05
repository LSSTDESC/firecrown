from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

import pyccl as ccl

from ..core import Source
from ..core import Systematic

__all__ = ["WLSource"]


@dataclass(frozen=True)
class WLSourceArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: np.ndarray
    dndz: np.ndarray
    ia_bias: np.ndarray


class WLSourceSystematic(Systematic):
    pass


def get_from_prefix_param(systematic: Systematic, params: Dict[str, float],
                          prefix: str,
                          param: str) -> float:
    p = None
    if prefix and f"{prefix}_{param}" in params.keys():
        p = params[f"{prefix}_{param}"]
    elif param in params.keys():
        p = params[param]
    else:
        typename = type(systematic).__name__
        raise KeyError(f"{typename} key `{param}' not found")

    return p


class MultiplicativeShearBias(WLSourceSystematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.


    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["mult_bias"]

    def __init__(self, sacc_tracer: str):
        """Create a MultipliciativeShearBias object that uses the named tracer.
        Parameters
        ----------
        sacc_tracer : The name of the multiplicative bias parameter.
        """
        self.sacc_tracer = sacc_tracer
        self.m: Optional[float] = None

    def update_params(self, params: Dict):
        """Read the corresponding named tracer from the given collection of parameters."""
        self.m = get_from_prefix_param(
            self, params, self.sacc_tracer, "mult_bias"
        )

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: WLSourceArgs):
        """Apply multiplicative shear bias to a source. The `scale_` of the
        source is multiplied by `(1 + m)`.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        tracer_arg : a WLSourceArgs object
            The WLSourceArgs to which apply the shear bias.
        """

        return WLSourceArgs(
            scale=tracer_arg.scale * (1.0 + self.m),
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            ia_bias=tracer_arg.ia_bias,
        )


class LinearAlignmentSystematic(WLSourceSystematic):
    """Linear alignment systematic.

    This systematic adds a linear intrinsic alignment model systematic
    which varies with redshift and the growth function.

    Parameters
    ----------
    alphaz :
        The redshift dependence parameter of the intrinsic alignment
        signal.
    alphag :
        The growth dependence parameter of the intrinsic alignment
        signal.
    z_piv :
        The pivot redshift parameter for the intrinsic alignment
        parameter.

    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["ia_bias", "alphaz", "alphag", "z_piv"]

    def __init__(self, sacc_tracer: Optional[str] = None):
        self.sacc_tracer = sacc_tracer

        self.ia_bias = None
        self.alphaz = None
        self.alphag = None
        self.z_piv = None

    def update_params(self, params):
        self.ia_bias = get_from_prefix_param(
            self, params, self.sacc_tracer, "ia_bias"
        )
        self.alphaz = get_from_prefix_param(
            self, params, self.sacc_tracer, "alphaz"
        )
        self.alphag = get_from_prefix_param(
            self, params, self.sacc_tracer, "alphag"
        )
        self.z_piv = get_from_prefix_param(
            self, params, self.sacc_tracer, "z_piv"
        )

    def apply(self, cosmo, tracer_arg: WLSourceArgs):
        """Apply a linear alignment systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        tracer_arg : a WLSourceArgs object
            The WLSourceArgs to which apply the shear bias.
        """

        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= ccl.growth_factor(cosmo, 1.0 / (1.0 + tracer_arg.z)) ** (
            self.alphag - 1.0
        )

        ia_bias_array = pref * self.ia_bias

        return WLSourceArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            ia_bias=(tracer_arg.z, ia_bias_array),
        )


class PhotoZShift(WLSourceSystematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.

    Parameters
    ----------
    delta_z : str
        The name of the photo-z shift parameter.

    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["delta_z"]

    def __init__(self, sacc_tracer: str):
        self.sacc_tracer = sacc_tracer
        self.delta_z = None

    def update_params(self, params):
        self.delta_z = get_from_prefix_param(
            self, params, self.sacc_tracer, "delta_z"
        )

    def apply(self, cosmo, tracer_arg: WLSourceArgs):
        """Apply a shift to the photo-z distribution of a source.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shift.
        """

        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)

        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0

        return WLSourceArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=dndz,
            ia_bias=tracer_arg.ia_bias,
        )


class WLSource(Source):
    def __init__(
        self,
        *,
        sacc_tracer,
        scale=1.0,
        systematics: Optional[List[WLSourceSystematic]] = None,
    ):
        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.z_orig: Optional[np.ndarray] = None
        self.dndz_orig: Optional[np.ndarray] = None
        self.dndz_interp = None

        self.systematics = []
        for systematic in systematics:
            self.systematics.append(systematic)

    def update_params(self, params):
        for systematic in self.systematics:
            systematic.update_params(params)

    def read(self, sacc_data):
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

        self.tracer_args = WLSourceArgs(scale=self.scale, z=z, dndz=nz, ia_bias=None)

    def render(self, cosmo, params, systematics=None):
        self.tracer_, tracer_args = self.create_tracer(cosmo, params, systematics)
        self.scale_ = tracer_args.scale

    def create_tracer(self, cosmo, params, systematics=None):
        """
        Render a source by applying systematics.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        systematics : dict
            A dictionary mapping systematic names to their objects. The
            default of `None` corresponds to no systematics.
        """
        systematics = systematics or {}

        tracer_args = self.tracer_args

        for systematic in self.systematics:
            tracer_args = systematic.apply(cosmo, tracer_args)

        tracer = ccl.WeakLensingTracer(
            cosmo, dndz=(tracer_args.z, tracer_args.dndz), ia_bias=tracer_args.ia_bias
        )

        return tracer, tracer_args
