"""Number counts source and systematics

"""

from __future__ import annotations
from typing import List, Optional, final
from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

from .source import Source, SourcePT
from .source import Systematic
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
    bias: np.ndarray
    mag_bias: np.ndarray

@dataclass(frozen=True)
class NumberCountsPTArgs:
    """Class for number counts tracer builder argument."""

    scale: float
    z: np.ndarray  # pylint: disable-msg=invalid-name
    dndz: np.ndarray
    bias: np.ndarray
    bias_2: np.ndarray
    bias_s: np.ndarray
    mag_bias: np.ndarray

class NumberCountsSystematic(Systematic):
    """Class implementing systematics for Number Counts sources."""

    @abstractmethod
    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply method to include systematics in the tracer_arg."""

class NumberCountsPTSystematic(Systematic):
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

        return NumberCountsArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=dndz,
            bias=tracer_arg.bias,
            mag_bias=tracer_arg.mag_bias,
        )


class NLBiasSystematic(NumberCountsPTSystematic):
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

    params_names = ["bias", "bias_2", "bias_s"]
    bias: float
    bias_2: float
    bias_s: float

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of
        parameters."""
        self.bias = params.get_from_prefix_param(self.sacc_tracer, "bias")
        self.bias_2 = params.get_from_prefix_param(self.sacc_tracer, "bias_2")
        self.bias_s = params.get_from_prefix_param(self.sacc_tracer, "bias_s")

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
        self, cosmo: pyccl.Cosmology, tracer_arg: NumberCountsPTArgs
    ) -> NumberCountsPTArgs:
        """Apply a linear bias systematic.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        tracer_arg : NumberCountsArgs
            The source to which apply the shear bias.
        """

        #pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        #pref *= pyccl.growth_factor(cosmo, 1.0 / (1.0 + tracer_arg.z)) ** self.alphag
        #print(tracer_arg.bias, tracer_arg.z)
        return NumberCountsPTArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            bias=tracer_arg.bias * np.ones_like(tracer_arg.z),
            bias_2=tracer_arg.bias_2 * np.ones_like(tracer_arg.z),
            bias_s=tracer_arg.bias_s * np.ones_like(tracer_arg.z),
            mag_bias=tracer_arg.mag_bias,
        )


class NumberCounts(Source):
    """Source class for number counts."""

    params_names = ["bias", "mag_bias"]
    bias: float
    mag_bias: Optional[float]

    systematics: UpdatableCollection
    tracer_arg: NumberCountsArgs

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
        self.has_mag_bias = has_mag_bias
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

        if self.has_mag_bias:
            self.mag_bias = params.get_from_prefix_param(self.sacc_tracer, "mag_bias")
        else:
            self.mag_bias = None

        self.systematics.update(params)

    @final
    def _reset_source(self) -> None:
        self.systematics.reset()

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


class NumberCountsPT(SourcePT):
    systematics: UpdatableCollection
    #tracer corresponds to the portion of the 'Source' which has no PT calculations, type Tracer
    #pttracer corresponds to the PTTracer used to calculate PT calculations
    #pt_tracer is a Tracer object that must be made for pyccl to calculate Cls from the PTTracer power spectrum

    tracer: Optional[pyccl.tracers.Tracer]
    pttracer: Optional[pyccl.nl_pt.tracers.PTTracer]
    pt_tracer: Optional[pyccl.tracers.Tracer]
    
    tracer_args: NumberCountsArgs
    pttracer_args: NumberCountsPTArgs
    pt_tracer_args: NumberCountsArgs
    
    cosmo_hash: Optional[int]
    ptcosmo_hash: Optional[int]
    pt_cosmo_hash: Optional[int]
    
    params_names = ["bias", "bias_2", "bias_s" "mag_bias"]
    bias: float
    mag_bias: Optional[float]

    systematics: UpdatableCollection
    tracer_arg: NumberCountsArgs

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
        self.has_mag_bias = has_mag_bias
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
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])
        derived_parameters = (
            derived_parameters + self.systematics.get_derived_parameters()
        )
        return derived_parameters
    
    @final
    def _update_source(self, params: ParamsMap):
        self.bias = params.get_from_prefix_param(self.sacc_tracer, "bias")
        self.bias_2 = params.get_from_prefix_param(self.sacc_tracer, "bias_2")
        self.bias_s = params.get_from_prefix_param(self.sacc_tracer, "bias_s")

        if self.has_mag_bias:
            self.mag_bias = params.get_from_prefix_param(self.sacc_tracer, "mag_bias")
        else:
            self.mag_bias = None

        self.ptcosmo_hash = None
        self.pttracer = None
        self.pt_cosmo_hash = None
        self.pt_tracer = None
        self.systematics.update(params)

    @final
    def _reset_source(self) -> None:
        self.systematics.reset()

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

        z = getattr(tracer, "z").copy().flatten()  # pylint: disable-msg=invalid-name
        nz = getattr(tracer, "nz").copy().flatten()  # pylint: disable-msg=invalid-name
        indices = np.argsort(z)
        z = z[indices]  # pylint: disable-msg=invalid-name
        nz = nz[indices]  # pylint: disable-msg=invalid-name

        self.pttracer_args = NumberCountsPTArgs(scale=self.scale, z=z, dndz=nz, bias=None, bias_2=None, bias_s=None)
        self.tracer_args = NumberCountsArgs(scale=self.scale, z=z, dndz=nz, bias=None)
        self.pt_tracer_args = NumberCountsArgs(scale=self.scale, z=z, dndz=nz, bias=None)



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

        self.pttracer_args = NumberCountsPTArgs(scale=self.scale, z=z, dndz=nz, bias=None, bias_2=None, bias_s=None,mag_bias=None)
        self.tracer_args = NumberCountsArgs(scale=self.scale, z=z, dndz=nz, bias=None, mag_bias=None)
        self.pt_tracer_args = NumberCountsArgs(scale=self.scale, z=z, dndz=nz, bias=None, mag_bias=None)


    def create_tracer(self, cosmo: pyccl.Cosmology):
        tracer_args = self.tracer_args

        bias = np.ones_like(tracer_args.z) * 0
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
            if not isinstance(systematic, NumberCountsPTSystematic):
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
    
    
    def create_pttracer(self, cosmo: pyccl.Cosmology):
        """
        Render a source by applying systematics.

        """
        pttracer_args = self.pttracer_args



        bias = np.ones_like(pttracer_args.z) * self.bias
        bias_2 = np.ones_like(pttracer_args.z) * self.bias_2
        bias_s = np.ones_like(pttracer_args.z) * self.bias_s
        pttracer_args = NumberCountsPTArgs(
            scale=pttracer_args.scale,
            z=pttracer_args.z,
            dndz=pttracer_args.dndz,
            bias=bias,
            bias_2 = bias_2,
            bias_s=bias_s,
            mag_bias=pttracer_args.mag_bias,
        )

        if self.mag_bias is not None:
            mag_bias = np.ones_like(pttracer_args.z) * self.mag_bias
            pttracer_args = NumberCountsPTArgs(
                scale=pttracer_args.scale,
                z=pttracer_args.z,
                dndz=pttracer_args.dndz,
                bias=pttracer_args.bias,
                bias_2 = pttracer_args.bias_2,
                bias_s=pttracer_args.bias_s,
                mag_bias=mag_bias,
            )

        for systematic in self.systematics:
            if  isinstance(systematic,NumberCountsPTSystematic):
                pttracer_args = systematic.apply(cosmo, pttracer_args)

        pttracer = pyccl.nl_pt.PTNumberCountsTracer( b1=(pttracer_args.z, pttracer_args.bias), b2 = (pttracer_args.z, pttracer_args.bias_2), bs=(pttracer_args.z, pttracer_args.bias_s))

        self.current_pttracer_args = pttracer_args

        return pttracer, pttracer_args

    def create_pt_tracer(self, cosmo: pyccl.Cosmology):
        """
        Render a source by applying systematics.

        """
        pttracer_args = self.pttracer_args
        pt_tracer_args = self.pt_tracer_args


        for systematic in self.systematics:
            if not isinstance(systematic,NumberCountsPTSystematic):
                pt_tracer_args = systematic.apply(cosmo, pt_tracer_args)

        pt_tracer = pyccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(pt_tracer_args.z, pt_tracer_args.dndz), bias=(pt_tracer_args.z, np.ones_like(pt_tracer_args.z)),mag_bias=None)

        self.current_pt_tracer_args = pt_tracer_args

        return pt_tracer, pt_tracer_args

    def get_pttracer(self, cosmo: pyccl.Cosmology) -> pyccl.nl_pt.tracers.PTTracer:
        """Return the tracer for the given cosmology.

        This method caches its result, so if called a second time with the same
        cosmology, no calculation needs to be done."""
        cur_hash = hash(cosmo)
        if hasattr(self, "ptcosmo_hash") and self.ptcosmo_hash == cur_hash:
            return self.pttracer

        self.pttracer, _ = self.create_pttracer(cosmo)
        self.ptcosmo_hash = cur_hash
        return self.pttracer
 
    def get_pt_tracer(self, cosmo: pyccl.Cosmology) -> pyccl.tracers.Tracer:
        """Return the tracer for the given cosmology.

        This method caches its result, so if called a second time with the same
        cosmology, no calculation needs to be done."""
        cur_hash = hash(cosmo)
        if hasattr(self, "pt_cosmo_hash") and self.pt_cosmo_hash == cur_hash:
            return self.pt_tracer

        self.pt_tracer, _ = self.create_pt_tracer(cosmo)
        self.pt_cosmo_hash = cur_hash
        return self.pt_tracer


    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale

    def get_ptscale(self):
        assert self.current_pttracer_args
        return self.current_pttracer_args.scale

    def get_pt_scale(self):
        assert self.current_pt_tracer_args
        return self.current_pt_tracer_args.scale