"""Weak lensing source and systematics

"""

from __future__ import annotations
from typing import List, Tuple, Optional, final
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
    ia_bias: Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class WeakLensingPTArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: np.ndarray  # pylint: disable-msg=invalid-name
    dndz: np.ndarray
    c1: Tuple[np.ndarray, np.ndarray]
    c2: Tuple[np.ndarray, np.ndarray]
    cdelta: Tuple[np.ndarray, np.ndarray]


class WeakLensingSystematic(Systematic):
    """Abstract base class for all weak lensing systematics."""

    @abstractmethod
    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply method to include systematics in the tracer_arg."""


class WeakLensingPTSystematic(Systematic):
    """Abstract base class for all weak lensing systematics."""

    @abstractmethod
    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingPTArgs
    ) -> WeakLensingPTArgs:
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
        """Create a MultipliciativeShearBias object that uses the named tracer.
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
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.ia_bias = params.get_from_prefix_param(self.sacc_tracer, "ia_bias")
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
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic, based on the given
        tracer_arg, in the context of the given cosmology."""

        pref = -(((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz)
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


class TATTSystematic(WeakLensingPTSystematic):
    """TATT alignment systematic.

    This systematic adds the TATT intrinsic alignment model systematic
    which varies with redshift and the growth function.

    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = [
        "ia_bias",
        "alphaz",
        "ia_bias_ta",
        "alphag",
        "z_piv",
        "ia_bias_2",
        "alphaz_2",
        "alphag_2",
    ]
    ia_bias: float
    alphaz: float
    alphag: float
    z_piv: float
    ia_bias_ta: float
    ia_bias_2: float
    alphaz_2: float
    alphag_2: float

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
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.ia_bias = params.get_from_prefix_param(self.sacc_tracer, "ia_bias")
        self.alphaz = params.get_from_prefix_param(self.sacc_tracer, "alphaz")
        self.alphag = params.get_from_prefix_param(self.sacc_tracer, "alphag")
        self.z_piv = params.get_from_prefix_param(self.sacc_tracer, "z_piv")
        self.ia_bias_ta = params.get_from_prefix_param(self.sacc_tracer, "ia_bias_ta")
        self.z_piv = params.get_from_prefix_param(self.sacc_tracer, "z_piv")
        self.ia_bias_2 = params.get_from_prefix_param(self.sacc_tracer, "ia_bias_2")
        self.alphaz_2 = params.get_from_prefix_param(self.sacc_tracer, "alphaz_2")
        self.alphag_2 = params.get_from_prefix_param(self.sacc_tracer, "alphag_2")

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
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingPTArgs
    ) -> WeakLensingPTArgs:
        c1_array, cdelta_array, c2_array = pyccl.nl_pt.translate_IA_norm(
            cosmo,
            tracer_arg.z,
            a1=self.ia_bias,
            a1delta=self.ia_bias_ta,
            a2=self.ia_bias_2,
            Om_m2_for_c2=False,
        )

        return WeakLensingPTArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            c1=(tracer_arg.z, c1_array),
            c2=(tracer_arg.z, c2_array),
            cdelta=(tracer_arg.z, cdelta_array),
        )


class PhotoZShift(WeakLensingSystematic):
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


class WeakLensingPT(SourcePT):
    # tracer corresponds to the portion of the 'Source' which
    # has no PT calculations, type Tracer
    # pttracer corresponds to the PTTracer used to calculate PT calculations
    # pt_tracer is a Tracer object that must be made for pyccl
    # to calculate Cls from the PTTracer power spectrum

    systematics: UpdatableCollection
    tracer: Optional[pyccl.tracers.Tracer]
    pttracer: Optional[pyccl.nl_pt.tracers.PTTracer]
    pt_tracer: Optional[pyccl.tracers.Tracer]
    tracer_args: WeakLensingArgs
    pttracer_args: WeakLensingPTArgs
    pt_tracer_args: WeakLensingArgs
    cosmo_hash: Optional[int]
    ptcosmo_hash: Optional[int]
    pt_cosmo_hash: Optional[int]

    def __init__(
        self,
        *,
        sacc_tracer: str,
        scale: float = 1.0,
        systematics: Optional[List[WeakLensingSystematic]] = None,
    ):
        """Initialize the WeakLensingPT object."""
        super().__init__()
        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.z_orig: Optional[np.ndarray] = None
        self.dndz_orig: Optional[np.ndarray] = None
        self.dndz_interp = None
        self.current_tracer_args: Optional[WeakLensingArgs] = None
        self.current_pttracer_args: Optional[WeakLensingPTArgs] = None
        self.current_pt_tracer_args: Optional[WeakLensingArgs] = None

        self.systematics = UpdatableCollection(systematics)

    @final
    def _update_source(self, params: ParamsMap):
        """Implementation of Source interface `_update_source`.

        This updates all the contained systematics."""
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

        self.pttracer_args = WeakLensingPTArgs(
            scale=self.scale, z=z, dndz=nz, c1=None, c2=None, cdelta=None
        )
        self.tracer_args = WeakLensingArgs(scale=self.scale, z=z, dndz=nz, ia_bias=None)
        self.pt_tracer_args = WeakLensingArgs(
            scale=self.scale, z=z, dndz=nz, ia_bias=None
        )

    def create_tracer(self, cosmo: pyccl.Cosmology):
        """
        Render a source by applying systematics.

        """
        tracer_args = self.tracer_args

        for systematic in self.systematics:
            if not isinstance(systematic, WeakLensingPTSystematic):
                tracer_args = systematic.apply(cosmo, tracer_args)

        tracer = pyccl.WeakLensingTracer(
            cosmo, dndz=(tracer_args.z, tracer_args.dndz), ia_bias=None, use_A_ia=False
        )

        self.current_tracer_args = tracer_args

        return tracer, tracer_args

    def create_pttracer(self, cosmo: pyccl.Cosmology):
        """
        Render a source by applying systematics.

        """
        pttracer_args = self.pttracer_args

        for systematic in self.systematics:
            if isinstance(systematic, WeakLensingPTSystematic):
                pttracer_args = systematic.apply(cosmo, pttracer_args)

        pttracer = pyccl.nl_pt.PTIntrinsicAlignmentTracer(
            c1=pttracer_args.c1, c2=pttracer_args.c2, cdelta=pttracer_args.cdelta
        )

        self.current_pttracer_args = pttracer_args

        return pttracer, pttracer_args

    def create_pt_tracer(self, cosmo: pyccl.Cosmology):
        """
        Render a source by applying systematics.

        """
        pt_tracer_args = self.pt_tracer_args

        for systematic in self.systematics:
            if not isinstance(systematic, WeakLensingPTSystematic):
                pt_tracer_args = systematic.apply(cosmo, pt_tracer_args)

        pt_tracer = pyccl.WeakLensingTracer(
            cosmo,
            has_shear=False,
            dndz=(pt_tracer_args.z, pt_tracer_args.dndz),
            ia_bias=(pt_tracer_args.z, np.ones_like(pt_tracer_args.z)),
            use_A_ia=False,
        )

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
