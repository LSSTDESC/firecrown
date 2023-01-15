"""Number counts source and systematics

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
from ..... import parameters
from .....parameters import (
    ParamsMap,
    RequiredParameters,
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

    """

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.alphaz = parameters.create()
        self.alphag = parameters.create()
        self.z_piv = parameters.create()
        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Perform any updates necessary after the parameters have being updated.

        This implementation has nothing to do."""

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

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
    r_lim : float
        The name of the limiting magnitude in r band filter.
    sig_c, eta, z_c, z_m : float
        The name of the fitting parameters in Joachimi & Bridle (2010) equation
        (C.1).

    Methods
    -------
    apply : apply the systematic to a source
    """

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.r_lim = parameters.create()
        self.sig_c = parameters.create()
        self.eta = parameters.create()
        self.z_c = parameters.create()
        self.z_m = parameters.create()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Perform any updates necessary after the parameters have being updated.

        This implementation has nothing to do."""

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

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
        # The slope of log(n_tot(z,r_lim)) with respect to r_lim
        # where n_tot(z,r_lim) is the luminosity function after using fit (C.1)
        # pylint: disable-next=invalid-name
        s = (
            self.eta / self.r_lim
            - 3.0 * self.z_m / z_bar
            + 1.5 * self.z_m * np.power(tracer_arg.z / z_bar, 1.5) / z_bar
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

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.delta_z = parameters.create()
        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Perform any updates necessary after the parameters have being updated.

        This implementation has nothing to do."""

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

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
    """Source class for number counts."""

    systematics: UpdatableCollection
    tracer_args: NumberCountsArgs

    def __init__(
        self,
        *,
        sacc_tracer: str,
        has_rsd: bool = False,
        has_mag_bias: bool = False,
        derived_scale: bool = False,
        scale: float = 1.0,
        systematics: Optional[List[NumberCountsSystematic]] = None,
    ):
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.has_rsd = has_rsd
        self.derived_scale = derived_scale

        self.bias = parameters.create()
        self.mag_bias = parameters.create(None if has_mag_bias else 0.0)
        self.systematics = UpdatableCollection(systematics)
        self.scale = scale
        self.current_tracer_args: Optional[NumberCountsArgs] = None

    @final
    def _update_source(self, params: ParamsMap):
        """Perform any updates necessary after the parameters have being updated.

        This implementation must update all contained Updatable instances."""
        self.systematics.update(params)

    @final
    def _reset_source(self) -> None:
        self.systematics.reset()

    @final
    def _required_parameters(self) -> RequiredParameters:
        return self.systematics.required_parameters()

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

        if self.mag_bias != 0.0:
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

        if self.mag_bias != 0.0:
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
