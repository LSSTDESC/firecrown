"""

Weak lensing source module
==========================

The classe in this file define ...

"""

from __future__ import annotations
from typing import List, Tuple, Optional, final
from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

from .source import Source
from .source import Systematic
from .....parameters import ParamsMap, RequiredParameters, parameter_get_full_name
from .....updatable import UpdatableCollection

__all__ = ["WeakLensing"]


@dataclass(frozen=True)
class WeakLensingArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: np.ndarray
    dndz: np.ndarray
    ia_bias: Tuple[np.ndarray, np.ndarray]


class WeakLensingSystematic(Systematic):
    @abstractmethod
    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply method to include systematics in the tracer_arg."""


class MultiplicativeShearBias(WeakLensingSystematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.


    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["mult_bias"]
    m: float

    def __init__(self, sacc_tracer: str):
        """Create a MultipliciativeShearBias object that uses the named tracer.
        Parameters
        ----------
        sacc_tracer : The name of the multiplicative bias parameter.
        """
        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of parameters."""
        self.m = params.get_from_prefix_param(self.sacc_tracer, "mult_bias")

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs):
        """Apply multiplicative shear bias to a source. The `scale_` of the
        source is multiplied by `(1 + m)`.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        tracer_arg : a WeakLensingArgs object
            The WeakLensingArgs to which apply the shear bias.
        """

        return WeakLensingArgs(
            scale=tracer_arg.scale * (1.0 + self.m),
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            ia_bias=tracer_arg.ia_bias,
        )


class LinearAlignmentSystematic(WeakLensingSystematic):
    """Linear alignment systematic.

    This systematic adds a linear intrinsic alignment model systematic
    which varies with redshift and the growth function.



    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["ia_bias", "alphaz", "alphag", "z_piv"]
    ia_bias: float
    alphaz: float
    alphag: float
    z_piv: float

    def __init__(self, sacc_tracer: Optional[str] = None):
        """Create a LinearAlignmentSystematic object, using the specified
        tracer name.

        Instance data are:

        alphaz : The redshift dependence parameter of the intrinsic alignment
        signal.

        alphag : The growth dependence parameter of the intrinsic alignment
        signal.

        z_piv : The pivot redshift parameter for the intrinsic alignment
        parameter.
        """
        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.ia_bias = params.get_from_prefix_param(self.sacc_tracer, "ia_bias")
        self.alphaz = params.get_from_prefix_param(self.sacc_tracer, "alphaz")
        self.alphag = params.get_from_prefix_param(self.sacc_tracer, "alphag")
        self.z_piv = params.get_from_prefix_param(self.sacc_tracer, "z_piv")

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic, based on the given
        tracer_arg, in the context of the given cosmology."""

        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= pyccl.growth_factor(cosmo, 1.0 / (1.0 + tracer_arg.z)) ** (
            self.alphag - 1.0
        )

        ia_bias_array = pref * self.ia_bias

        return WeakLensingArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            ia_bias=(tracer_arg.z, ia_bias_array),
        )


class PhotoZShift(WeakLensingSystematic):
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

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs):
        """Apply a shift to the photo-z distribution of a source."""

        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)

        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0

        return WeakLensingArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=dndz,
            ia_bias=tracer_arg.ia_bias,
        )


class WeakLensing(Source):
    systematics: UpdatableCollection
    tracer_arg: WeakLensingArgs

    def __init__(
        self,
        *,
        sacc_tracer,
        scale=1.0,
        systematics: Optional[List[WeakLensingSystematic]] = None,
    ):
        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.z_orig: Optional[np.ndarray] = None
        self.dndz_orig: Optional[np.ndarray] = None
        self.dndz_interp = None
        self.current_tracer_args = None
        self.scale_ = None
        self.tracer_ = None

        self.systematics = UpdatableCollection([])
        if systematics:
            for systematic in systematics:
                self.systematics.append(systematic)

    @final
    def _update_source(self, params: ParamsMap):
        self.systematics.update(params)

    @final
    def required_parameters(self) -> RequiredParameters:
        return self.systematics.required_parameters()

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

        self.tracer_args = WeakLensingArgs(scale=self.scale, z=z, dndz=nz, ia_bias=None)

    def create_tracer(self, cosmo: pyccl.Cosmology):
        """
        Render a source by applying systematics.

        """
        tracer_args = self.tracer_args

        for systematic in self.systematics:
            tracer_args = systematic.apply(cosmo, tracer_args)

        tracer = pyccl.WeakLensingTracer(
            cosmo, dndz=(tracer_args.z, tracer_args.dndz), ia_bias=tracer_args.ia_bias
        )
        self.current_tracer_args = tracer_args

        return tracer, tracer_args

    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale
