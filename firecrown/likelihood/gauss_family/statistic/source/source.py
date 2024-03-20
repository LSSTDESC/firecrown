"""Abstract base classes for TwoPoint Statistics sources.
"""

from __future__ import annotations
from typing import Optional, Sequence, final, TypeVar, Generic
from abc import abstractmethod
from dataclasses import dataclass, replace
import numpy as np
import numpy.typing as npt
from scipy.interpolate import Akima1DInterpolator

import sacc

import pyccl
import pyccl.nl_pt

from .....modeling_tools import ModelingTools
from .....parameters import ParamsMap
from ..... import parameters
from .....updatable import Updatable, UpdatableCollection


class SourceSystematic(Updatable):
    """An abstract systematic class (e.g., shear biases, photo-z shifts, etc.).

    This class currently has no methods at all, because the argument types for
    the `apply` method of different subclasses are different."""

    def read(self, sacc_data: sacc.Sacc):
        """This method is called to allow the systematic object to read from the
        appropriated sacc data."""


class Source(Updatable):
    """An abstract source class (e.g., a sample of lenses).

    Parameters
    ----------
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.
    """

    systematics: Sequence[SourceSystematic]
    cosmo_hash: Optional[int]
    tracers: Sequence[Tracer]

    def __init__(self, sacc_tracer: str) -> None:
        """Create a Source object that uses the named tracer.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)
        self.sacc_tracer = sacc_tracer

    @final
    def read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file."""
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.read(sacc_data)
        self._read(sacc_data)

    @abstractmethod
    def _read(self, sacc_data: sacc.Sacc):
        """Abstract method to read the data for this source from the SACC file."""

    def _update_source(self, params: ParamsMap):
        """Method to update the source from the given ParamsMap. Any subclass
        that needs to do more than update its contained :class:`Updatable`
        objects should implement this method."""

    @final
    def _update(self, params: ParamsMap):
        """Implementation of Updatable interface method `_update`.

        This clears the current hash and tracer, and calls the abstract method
        `_update_source`, which must be implemented in all subclasses."""
        self.cosmo_hash = None
        self.tracers = []
        self._update_source(params)

    @abstractmethod
    def get_scale(self) -> float:
        """Abstract method to return the scales for this `Source`."""

    @abstractmethod
    def create_tracers(self, tools: ModelingTools):
        """Abstract method to create tracers for this `Source`, for the given
        cosmology."""

    @final
    def get_tracers(self, tools: ModelingTools) -> Sequence[Tracer]:
        """Return the tracer for the given cosmology.

        This method caches its result, so if called a second time with the same
        cosmology, no calculation needs to be done."""

        ccl_cosmo = tools.get_ccl_cosmology()

        cur_hash = hash(ccl_cosmo)
        if hasattr(self, "cosmo_hash") and self.cosmo_hash == cur_hash:
            return self.tracers

        self.tracers, _ = self.create_tracers(tools)
        self.cosmo_hash = cur_hash
        return self.tracers


class Tracer:
    """Bundles together a pyccl.Tracer object with optional information about the
    underlying 3D field, a pyccl.nl_pt.PTTracer, and halo profiles."""

    @staticmethod
    def determine_field_name(field: Optional[str], tracer: Optional[str]) -> str:
        """This function encapsulates the policy for determining the value to be
        assigned to the :attr:`field` attribute of a :class:`Tracer`.

        It is a static method only to keep it grouped with the class for which it is
        defining the initialization policy.
        """
        if field is not None:
            return field
        if tracer is not None:
            return tracer
        return "delta_matter"

    def __init__(
        self,
        tracer: pyccl.Tracer,
        tracer_name: Optional[str] = None,
        field: Optional[str] = None,
        pt_tracer: Optional[pyccl.nl_pt.PTTracer] = None,
        halo_profile: Optional[pyccl.halos.HaloProfile] = None,
        halo_2pt: Optional[pyccl.halos.Profile2pt] = None,
    ):
        """Initialize a new Tracer based on the given pyccl.Tracer which must not be
        None.

        Note that the :class:`pyccl.Tracer` is not copied; we store a reference to the
        original tracer. Be careful not to accidentally share :class:`pyccl.Tracer`s.

        If no tracer_name is supplied, then the tracer_name is set to the name of the
        :class:`pyccl.Tracer` class that was used.

        If no `field` is given, then the attribute :attr:`field` is set to either
        (1) the tracer_name, if one was given, or (2) 'delta_matter'.
        """
        assert tracer is not None
        self.ccl_tracer = tracer
        self.tracer_name: str = tracer_name or tracer.__class__.__name__
        self.field = Tracer.determine_field_name(field, tracer_name)
        self.pt_tracer = pt_tracer
        self.halo_profile = halo_profile
        self.halo_2pt = halo_2pt

    @property
    def has_pt(self) -> bool:
        """Return True if we have a pt_tracer, and False if not."""
        return self.pt_tracer is not None

    @property
    def has_hm(self) -> bool:
        """Return True if we have a halo_profile, and False if not."""
        return self.halo_profile is not None


# Sources of galaxy distributions


@dataclass(frozen=True)
class SourceGalaxyArgs:
    """Class for galaxy based sources arguments."""

    z: npt.NDArray[np.float64]
    dndz: npt.NDArray[np.float64]

    scale: float = 1.0

    field: str = "delta_matter"


_SourceGalaxyArgsT = TypeVar("_SourceGalaxyArgsT", bound=SourceGalaxyArgs)


class SourceGalaxySystematic(SourceSystematic, Generic[_SourceGalaxyArgsT]):
    """Abstract base class for all galaxy based source systematics."""

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: _SourceGalaxyArgsT
    ) -> _SourceGalaxyArgsT:
        """Apply method to include systematics in the tracer_arg."""


_SourceGalaxySystematicT = TypeVar(
    "_SourceGalaxySystematicT", bound=SourceGalaxySystematic
)


class SourceGalaxyPhotoZShift(
    SourceGalaxySystematic[_SourceGalaxyArgsT], Generic[_SourceGalaxyArgsT]
):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some amount `delta_z`.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar delta_z: the photo-z shift.
    """

    def __init__(self, sacc_tracer: str) -> None:
        """Create a PhotoZShift object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.delta_z = parameters.register_new_updatable_parameter()

    def apply(self, tools: ModelingTools, tracer_arg: _SourceGalaxyArgsT):
        """Apply a shift to the photo-z distribution of a source."""

        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)

        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0

        return replace(
            tracer_arg,
            dndz=dndz,
        )


class SourceGalaxySelectField(
    SourceGalaxySystematic[_SourceGalaxyArgsT], Generic[_SourceGalaxyArgsT]
):
    """A systematic that allows specifying the 3D field that will be used
    to select the 3D power spectrum when computing the angular power
    spectrum.
    """

    def __init__(self, field: str = "delta_matter"):
        """Specify which 3D field should be used when computing angular power
        spectra.

        :param field: the name of the 3D field that is associated to the tracer.
            Default: `"delta_matter"`
        """
        super().__init__()
        self.field = field

    def apply(
        self, tools: ModelingTools, tracer_arg: _SourceGalaxyArgsT
    ) -> _SourceGalaxyArgsT:
        return replace(tracer_arg, field=self.field)


class SourceGalaxy(Source, Generic[_SourceGalaxyArgsT]):
    """Source class for galaxy based sources."""

    def __init__(
        self,
        *,
        sacc_tracer: str,
        systematics: Optional[list[SourceGalaxySystematic]] = None,
    ):
        """Initialize the SourceGalaxy object.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(sacc_tracer)

        self.sacc_tracer = sacc_tracer
        self.current_tracer_args: Optional[_SourceGalaxyArgsT] = None
        self.systematics: UpdatableCollection[
            SourceGalaxySystematic
        ] = UpdatableCollection(systematics)
        self.tracer_args: _SourceGalaxyArgsT

    def _read(self, sacc_data: sacc.Sacc):
        """Read the galaxy redshift distribution model from a sacc file.
        All derived classes must call this method in their own `_read` method
        after they have read their own data and initialized their tracer_args."""

        try:
            tracer_args = self.tracer_args
        except AttributeError as exc:
            raise RuntimeError(
                "Must initialize tracer_args before calling _read on SourceGalaxy"
            ) from exc

        tracer = sacc_data.get_tracer(self.sacc_tracer)

        z = getattr(tracer, "z").copy().flatten()
        nz = getattr(tracer, "nz").copy().flatten()
        indices = np.argsort(z)
        z = z[indices]
        nz = nz[indices]

        self.tracer_args = replace(
            tracer_args,
            z=z,
            dndz=nz,
        )
