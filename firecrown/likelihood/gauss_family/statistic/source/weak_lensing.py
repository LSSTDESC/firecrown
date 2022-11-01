"""Weak lensing source and systematics

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
from ..... import parameters
from .....parameters import (
    ParamsMap,
    RequiredParameters,
    parameter_get_full_name,
    DerivedParameterCollection,
)
from .....updatable import UpdatableCollection

__all__ = ["WeakLensing"]


@dataclass(frozen=True)
class WeakLensingArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: np.ndarray  # pylint: disable-msg=invalid-name
    dndz: np.ndarray
    ia_bias: Optional[Tuple[np.ndarray, np.ndarray]]


class WeakLensingSystematic(Systematic):
    """Abstract base class for all weak lensing systematics."""

    @abstractmethod
    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply method to include systematics in the tracer_arg."""


class MultiplicativeShearBias(WeakLensingSystematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.

    Parameters
    ----------
    mult_bias : str
       The name of the multiplicative bias parameter.
    """

    params_names = ["mult_bias"]
    m: float

    def __init__(self, sacc_tracer: str):
        """Create a MultiplicativeShearBias object that uses the named tracer.
        Parameters
        ----------
        sacc_tracer : The name of the multiplicative bias parameter.
        """
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of
        parameters."""
        # pylint: disable-next=invalid-name
        self.m = params.get_from_prefix_param(self.sacc_tracer, "mult_bias")

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

    def __init__(self, sacc_tracer: Optional[str] = None, alphag=1.0):
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
        super().__init__()

        self.ia_bias = parameters.create()
        self.alphaz = parameters.create()
        self.alphag = parameters.create(alphag)
        self.z_piv = parameters.create()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        pass

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic, based on the given
        tracer_arg, in the context of the given cosmology."""

        alphag = self.alphag
        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= pyccl.growth_factor(cosmo, 1.0 / (1.0 + tracer_arg.z)) ** (
            alphag - 1.0
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

    This systematic shifts the photo-z distribution by some amount `delta_z`.
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
        pass

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])

        return derived_parameters

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
    """Source class for weak lensing."""

    systematics: UpdatableCollection
    tracer_args: WeakLensingArgs

    def __init__(
        self,
        *,
        sacc_tracer: str,
        scale: float = 1.0,
        systematics: Optional[List[WeakLensingSystematic]] = None,
    ):
        """Initialize the WeakLensing object."""
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.z_orig: Optional[np.ndarray] = None
        self.dndz_orig: Optional[np.ndarray] = None
        self.dndz_interp = None
        self.current_tracer_args: Optional[WeakLensingArgs] = None
        self.systematics = UpdatableCollection(systematics)

    @final
    def _update_source(self, params: ParamsMap):
        """Implementation of Source interface `_update_source`.

        This updates all the contained systematics."""
        self.systematics.update(params)

    @final
    def _reset_source(self) -> None:
        self.systematics.reset()

    @final
    def required_parameters(self) -> RequiredParameters:
        return self.systematics.required_parameters()

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
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

        z = getattr(tracer, "z").copy().flatten()  # pylint: disable-msg=invalid-name
        nz = getattr(tracer, "nz").copy().flatten()  # pylint: disable-msg=invalid-name
        indices = np.argsort(z)
        z = z[indices]  # pylint: disable-msg=invalid-name
        nz = nz[indices]  # pylint: disable-msg=invalid-name

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
