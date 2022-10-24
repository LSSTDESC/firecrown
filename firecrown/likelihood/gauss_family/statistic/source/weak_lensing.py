"""Weak lensing source and systematics

"""

from __future__ import annotations
from typing import List, Tuple, Optional, final
from dataclasses import dataclass, replace
from abc import abstractmethod

import numpy as np
import pyccl
import pyccl.nl_pt
from scipy.interpolate import Akima1DInterpolator

from .source import Source, TracerBundle
from .source import Systematic
from .....parameters import (
    ParamsMap,
    RequiredParameters,
    parameter_get_full_name,
    DerivedParameterCollection,
)
from .....likelihood.likelihood import CosmologyBundle

from .....updatable import UpdatableCollection

__all__ = ["WeakLensing"]


@dataclass(frozen=True)
class WeakLensingArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: np.ndarray  # pylint: disable-msg=invalid-name
    dndz: np.ndarray
    ia_bias: Tuple[np.ndarray, np.ndarray]

    has_pt: bool = False
    has_hm: bool = False

    ia_pt_c_1: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ia_pt_c_d: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ia_pt_c_2: Optional[Tuple[np.ndarray, np.ndarray]] = None


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


class TattAlignmentSystematic(WeakLensingSystematic):
    """TATT alignment systematic.

    This systematic adds a TATT (nonlinear) intrinsic alignment model systematic.
    Parameters can vary with redshift and the growth function.
    This also requires PTTracers and a PTCalculator.

    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["ia_a_1", "ia_a_2", "ia_a_d"]
    a_1: float
    a_2: float
    a_d: float

    def __init__(self, sacc_tracer: Optional[str] = None):
        """Create a TattAlignmentSystematic object, using the specified
        tracer name.

        Instance data are:

        [UPDATE THESE PARAMETERS]

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
        self.a_1 = params.get_from_prefix_param(self.sacc_tracer, "ia_a_1")
        self.a_2 = params.get_from_prefix_param(self.sacc_tracer, "ia_a_2")
        self.a_d = params.get_from_prefix_param(self.sacc_tracer, "ia_a_d")

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
        self, cosmo: CosmologyBundle, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic, based on the given
        tracer_arg, in the context of the given cosmology."""
        z = tracer_arg.z
        c_1, c_d, c_2 = pyccl.nl_pt.translate_IA_norm(
            cosmo.ccl_cosmo,
            z,
            a1=self.a_1,
            a1delta=self.a_d,
            a2=self.a_2,
            Om_m2_for_c2=False,
        )

        return replace(
            tracer_arg,
            has_pt=True,
            ia_pt_c_1=(z, c_1),
            ia_pt_c_d=(z, c_d),
            ia_pt_c_2=(z, c_2),
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

    def create_tracers(self, cosmo: CosmologyBundle):
        """
        Render a source by applying systematics.

        """
        tracer_args = self.tracer_args

        for systematic in self.systematics:
            tracer_args = systematic.apply(cosmo, tracer_args)

        wl_tracer = pyccl.WeakLensingTracer(
            cosmo.ccl_cosmo,
            dndz=(tracer_args.z, tracer_args.dndz),
            ia_bias=tracer_args.ia_bias,
        )
        tracer_containers = [TracerBundle(wl_tracer, field="delta_matter")]

        if tracer_args.has_pt:
            ia_pt_tracer = pyccl.nl_pt.PTIntrinsicAlignmentTracer(
                c1=tracer_args.ia_pt_c_1,
                cdelta=tracer_args.ia_pt_c_d,
                c2=tracer_args.ia_pt_c_2,
            )
            matter_ia_pt_tracer = pyccl.nl_pt.PTMatterTracer()

            wl_dummy_tracer = pyccl.WeakLensingTracer(
                cosmo.ccl_cosmo,
                has_shear=False,
                use_A_ia=False,
                dndz=(tracer_args.z, tracer_args.dndz),
                ia_bias=(tracer_args.z, np.ones_like(tracer_args.z)),
            )
            ia_tracer_container = TracerBundle(
                wl_dummy_tracer, field="intrinsic_pt", pt_tracer=ia_pt_tracer
            )
            matter_pt_tracer_container = TracerBundle(
                wl_tracer, field="delta_matter", pt_tracer=matter_ia_pt_tracer
            )
            tracer_containers.append(ia_tracer_container)
            tracer_containers.append(matter_pt_tracer_container)

        self.current_tracer_args = tracer_args

        return tracer_containers, tracer_args

    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale
