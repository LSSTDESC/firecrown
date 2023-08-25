"""Weak lensing source and systematics

"""

from __future__ import annotations
from typing import List, Tuple, Optional, final
from dataclasses import dataclass, replace
from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.nl_pt
import sacc
from scipy.interpolate import Akima1DInterpolator

from .source import Source, Tracer, SourceSystematic
from ..... import parameters
from .....parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
)
from .....modeling_tools import ModelingTools
from .....updatable import UpdatableCollection

__all__ = ["WeakLensing"]


@dataclass(frozen=True)
class WeakLensingArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: npt.NDArray[np.float64]
    dndz: npt.NDArray[np.float64]
    ia_bias: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]

    has_pt: bool = False
    has_hm: bool = False

    ia_pt_c_1: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    ia_pt_c_d: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    ia_pt_c_2: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None


class WeakLensingSystematic(SourceSystematic):
    """Abstract base class for all weak lensing systematics."""

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply method to include systematics in the tracer_arg."""


class MultiplicativeShearBias(WeakLensingSystematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.

    """

    def __init__(self, sacc_tracer: str) -> None:
        """Create a MultiplicativeShearBias object that uses the named tracer.
        Parameters
        ----------
        sacc_tracer : The name of the multiplicative bias parameter.
        """
        super().__init__()

        self.mult_bias = parameters.create()
        self.sacc_tracer = sacc_tracer

    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply multiplicative shear bias to a source. The `scale_` of the
        source is multiplied by `(1 + m)`.

        :param tools: A ModelingTools object.
        :param tracer_arg: The WeakLensingArgs to which apply the shear bias.

        :returns: A new WeakLensingArgs object with the shear bias applied.
        """

        return replace(
            tracer_arg,
            scale=tracer_arg.scale * (1.0 + self.mult_bias),
        )


class LinearAlignmentSystematic(WeakLensingSystematic):
    """Linear alignment systematic.

    This systematic adds a linear intrinsic alignment model systematic
    which varies with redshift and the growth function.

    Methods
    -------
    apply : apply the systematic to a source
    """

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
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic, based on the given
        tracer_arg, in the context of the given cosmology."""

        ccl_cosmo = tools.get_ccl_cosmology()

        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= pyccl.growth_factor(ccl_cosmo, 1.0 / (1.0 + tracer_arg.z)) ** (
            self.alphag - 1.0
        )

        ia_bias_array = pref * self.ia_bias

        return replace(
            tracer_arg,
            ia_bias=(tracer_arg.z, ia_bias_array),
        )


class TattAlignmentSystematic(WeakLensingSystematic):
    """TATT alignment systematic.

    This systematic adds a TATT (nonlinear) intrinsic alignment model systematic.

    Parameters
    ----------
    ia_a_1: float
    ia_a_2: float
    ia_a_d: float

    Methods
    -------
    apply : apply the systematic to a source
    """

    def __init__(self, sacc_tracer: Optional[str] = None):
        super().__init__()
        self.ia_a_1 = parameters.create()
        self.ia_a_2 = parameters.create()
        self.ia_a_d = parameters.create()

        self.sacc_tracer = sacc_tracer

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic, based on the given
        tracer_arg, in the context of the given cosmology."""

        ccl_cosmo = tools.get_ccl_cosmology()
        z = tracer_arg.z
        c_1, c_d, c_2 = pyccl.nl_pt.translate_IA_norm(
            ccl_cosmo,
            z=z,
            a1=self.ia_a_1,
            a1delta=self.ia_a_d,
            a2=self.ia_a_2,
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

    This systematic shifts the photo-z distribution by some amount `delta_z`.
    """

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.delta_z = parameters.create()
        self.sacc_tracer = sacc_tracer

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])

        return derived_parameters

    def apply(self, tools: ModelingTools, tracer_arg: WeakLensingArgs):
        """Apply a shift to the photo-z distribution of a source."""

        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)

        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0

        return replace(
            tracer_arg,
            dndz=dndz,
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
        self.current_tracer_args: Optional[WeakLensingArgs] = None
        self.systematics = UpdatableCollection(systematics)

    @final
    def _update_source(self, params: ParamsMap):
        """Implementation of Source interface `_update_source`.

        This updates all the contained systematics."""
        self.systematics.update(params)

    @final
    def _required_parameters(self) -> RequiredParameters:
        return self.systematics.required_parameters()

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])
        derived_parameters = (
            derived_parameters + self.systematics.get_derived_parameters()
        )
        return derived_parameters

    def _read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this source from the SACC file.

        This sets self.tracer_args, based on the data in `sacc_data` associated with
        this object's `sacc_tracer` name.
        """
        tracer = sacc_data.get_tracer(self.sacc_tracer)

        z = getattr(tracer, "z").copy().flatten()
        nz = getattr(tracer, "nz").copy().flatten()
        indices = np.argsort(z)
        z = z[indices]
        nz = nz[indices]

        self.tracer_args = WeakLensingArgs(scale=self.scale, z=z, dndz=nz, ia_bias=None)

    def create_tracers(self, tools: ModelingTools):
        """
        Render a source by applying systematics.

        """

        ccl_cosmo = tools.get_ccl_cosmology()
        tracer_args = self.tracer_args

        assert self.systematics is not None
        for systematic in self.systematics:
            tracer_args = systematic.apply(tools, tracer_args)

        ccl_wl_tracer = pyccl.WeakLensingTracer(
            ccl_cosmo,
            dndz=(tracer_args.z, tracer_args.dndz),
            ia_bias=tracer_args.ia_bias,
        )
        tracers = [Tracer(ccl_wl_tracer, tracer_name="shear", field="delta_matter")]

        if tracer_args.has_pt:
            ia_pt_tracer = pyccl.nl_pt.PTIntrinsicAlignmentTracer(
                c1=tracer_args.ia_pt_c_1,
                cdelta=tracer_args.ia_pt_c_d,
                c2=tracer_args.ia_pt_c_2,
            )

            ccl_wl_dummy_tracer = pyccl.WeakLensingTracer(
                ccl_cosmo,
                has_shear=False,
                use_A_ia=False,
                dndz=(tracer_args.z, tracer_args.dndz),
                ia_bias=(tracer_args.z, np.ones_like(tracer_args.z)),
            )
            ia_tracer = Tracer(
                ccl_wl_dummy_tracer, tracer_name="intrinsic_pt", pt_tracer=ia_pt_tracer
            )
            tracers.append(ia_tracer)

        self.current_tracer_args = tracer_args

        return tracers, tracer_args

    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale
