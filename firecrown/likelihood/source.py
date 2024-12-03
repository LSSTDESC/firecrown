"""Abstract base classes for TwoPoint Statistics sources."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, Sequence, TypeVar, final, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.nl_pt
import sacc
from scipy.interpolate import Akima1DInterpolator

# firecrown is needed for backward compatibility; remove support for deprecated
# directory structure is removed.
import firecrown  # pylint: disable=unused-import # noqa: F401
from firecrown import parameters
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from firecrown.updatable import Updatable, UpdatableCollection


class SourceSystematic(Updatable):
    """An abstract systematic class (e.g., shear biases, photo-z shifts, etc.).

    This class currently has no methods at all, because the argument types for
    the `apply` method of different subclasses are different.
    """

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Call to allow this object to read from the appropriate sacc data.

        :param sacc_data: The SACC data object to be read
        """


class Source(Updatable):
    """The abstract base class for all sources."""

    systematics: Sequence[SourceSystematic]
    cosmo_hash: None | int
    tracers: Sequence[Tracer]

    def __init__(self, sacc_tracer: str) -> None:
        """Create a Source object that uses the named tracer.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)
        self.sacc_tracer = sacc_tracer

    @final
    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this source from the SACC file.

        :param sacc_data: The SACC data object to be read
        """
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.read(sacc_data)
        self._read(sacc_data)

    @abstractmethod
    def _read(self, sacc_data: sacc.Sacc) -> None:
        """Abstract method to read the data for this source from the SACC file.

        :param sacc_data: The SACC data object to be read
        """

    def _update_source(self, params: ParamsMap) -> None:
        """Method to update the source from the given ParamsMap.

        Any subclass that needs to do more than update its contained :class:`Updatable`
        objects should implement this method.

        :param params: the parameters to be used for the update
        """

    @final
    def _update(self, params: ParamsMap):
        """Implementation of Updatable interface method `_update`.

        This clears the current hash and tracer, and calls the abstract method
        `_update_source`, which must be implemented in all subclasses.

        :param params: the parameters to be used for the update
        """
        self.cosmo_hash = None
        self.tracers = []
        self._update_source(params)

    @abstractmethod
    def get_scale(self) -> float:
        """Abstract method to return the scale for this `Source`.

        :return: the scale
        """

    @abstractmethod
    def create_tracers(self, tools: ModelingTools):
        """Abstract method to create tracers for this Source.

        :param tools: The modeling tools used for creating the tracers
        """

    @final
    def get_tracers(self, tools: ModelingTools) -> Sequence[Tracer]:
        """Return the tracer for the given cosmology.

        This method caches its result, so if called a second time with the same
        cosmology, no calculation needs to be done.

        :param tools: The modeling tools used for creating the tracers
        :return: the list of tracers
        """
        ccl_cosmo = tools.get_ccl_cosmology()

        cur_hash = hash(ccl_cosmo)
        if hasattr(self, "cosmo_hash") and self.cosmo_hash == cur_hash:
            return self.tracers

        self.tracers, _ = self.create_tracers(tools)
        self.cosmo_hash = cur_hash
        return self.tracers


class Tracer:
    """Extending the pyccl.Tracer object with additional information.

    Bundles together a pyccl.Tracer object with optional information about the
    underlying 3D field, or a pyccl.nl_pt.PTTracer.
    """

    @staticmethod
    def determine_field_name(field: None | str, tracer: None | str) -> str:
        """Gets a field name for a tracer.

        This function encapsulates the policy for determining the value to be
        assigned to the :attr:`field` attribute of a :class:`Tracer`.

        It is a static method only to keep it grouped with the class for which it is
        defining the initialization policy.

        :param field: the (stub) name of the field
        :param tracer: the name of the tracer
        :return: the full name of the field
        """
        if field is not None:
            return field
        if tracer is not None:
            return tracer
        return "delta_matter"

    def __init__(
        self,
        tracer: pyccl.Tracer,
        tracer_name: None | str = None,
        field: None | str = None,
        pt_tracer: None | pyccl.nl_pt.PTTracer = None,
    ):
        """Initialize a new Tracer based on the provided tracer.

        Note that the :class:`pyccl.Tracer` is not copied; we store a reference to the
        original tracer. Be careful not to accidentally share :class:`pyccl.Tracer`s.

        If no tracer_name is supplied, then the tracer_name is set to the name of the
        :class:`pyccl.Tracer` class that was used.

        If no `field` is given, then the attribute :attr:`field` is set to either
        (1) the tracer_name, if one was given, or (2) 'delta_matter'.

        :param tracer: the pyccl.Tracer used as the basis for this Tracer.
        :param tracer_name: optional name of the tracer.
        :param field: optional name of the field associated with the tracer.
        :param pt_tracer: optional non-linear perturbation theory tracer.
        """
        assert tracer is not None
        self.ccl_tracer = tracer
        self.tracer_name: str = tracer_name or tracer.__class__.__name__
        self.field = Tracer.determine_field_name(field, tracer_name)
        self.pt_tracer = pt_tracer

    @property
    def has_pt(self) -> bool:
        """Answer whether we have a perturbation theory tracer.

        :return: True if we have a pt_tracer, and False if not.
        """
        return self.pt_tracer is not None


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
    """Abstract base class for all galaxy-based source systematics."""

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: _SourceGalaxyArgsT
    ) -> _SourceGalaxyArgsT:
        """Apply method to include systematics in the tracer_arg.

        :param tools: the modeling tools use to update the tracer arg
        :param tracer_arg: the original source galaxy tracer arg to which we
           apply the systematic.
        :return: a new source galaxy tracer arg with the systematic applied
        """


_SourceGalaxySystematicT = TypeVar(
    "_SourceGalaxySystematicT", bound=SourceGalaxySystematic
)


SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z = 0.0
SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z = 1.0


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

        self.delta_z = parameters.register_new_updatable_parameter(
            default_value=SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: _SourceGalaxyArgsT
    ) -> _SourceGalaxyArgsT:
        """Apply a shift to the photo-z distribution of a source.

        :param tools: the modeling tools use to update the tracer arg
        :param tracer_arg: the original source galaxy tracer arg to which we
            apply the systematic.
        :return: a new source galaxy tracer arg with the systematic applied
        """
        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)

        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0

        return replace(
            tracer_arg,
            dndz=dndz,
        )


class PhotoZShift(SourceGalaxyPhotoZShift):
    """Photo-z shift systematic."""


class PhotoZShiftFactory(BaseModel):
    """Factory class for PhotoZShift objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["PhotoZShiftFactory"],
        Field(description="The type of the systematic."),
    ] = "PhotoZShiftFactory"

    def create(self, bin_name: str) -> PhotoZShift:
        """Create a PhotoZShift object with the given tracer name."""
        return PhotoZShift(bin_name)

    def create_global(self) -> PhotoZShift:
        """Create a PhotoZShift object with the given tracer name."""
        raise ValueError("PhotoZShift cannot be global.")


class SourceGalaxyPhotoZShiftandStretch(SourceGalaxyPhotoZShift[_SourceGalaxyArgsT]):
    """A photo-z shift & stretch bias.

    This systematic shifts and widens the photo-z distribution by some amount `delta_z`.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar delta_z: the photo-z shift.
    :ivar sigma_z: the photo-z stretch.
    """

    def __init__(self, sacc_tracer: str) -> None:
        """Create a PhotoZShift object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(sacc_tracer)

        self.sigma_z = parameters.register_new_updatable_parameter(
            default_value=SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z
        )

    def apply(self, tools: ModelingTools, tracer_arg: _SourceGalaxyArgsT):
        """Apply a shift & stretch to the photo-z distribution of a source."""
        super().apply(tools, tracer_arg)
        z = tracer_arg.z
        dndz = tracer_arg.dndz
        dndz_interp = Akima1DInterpolator(z, dndz)
        dndz_mean = np.average(tracer_arg.z, weights=tracer_arg.dndz)
        if self.sigma_z <= 0.0:
            raise ValueError("Stretch Parameter must be positive")
        dndz = (
            dndz_interp((z - dndz_mean) / self.sigma_z + dndz_mean, extrapolate=False)
            / self.sigma_z
        )
        # This is dangerous
        dndz[np.isnan(dndz)] = 0.0

        return replace(
            tracer_arg,
            dndz=dndz,
        )


class PhotoZShiftandStretch(SourceGalaxyPhotoZShiftandStretch):
    """Photo-z shift and stretch systematic."""


class PhotoZShiftandStretchFactory(BaseModel):
    """Factory class for PhotoZShiftandStretch objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["PhotoZShiftandStretchFactory"],
        Field(description="The type of the systematic."),
    ] = "PhotoZShiftandStretchFactory"

    def create(self, bin_name: str) -> PhotoZShiftandStretch:
        """Create a PhotoZShiftandStretch object with the given tracer name."""
        return PhotoZShiftandStretch(bin_name)

    def create_global(self) -> PhotoZShiftandStretch:
        """Create a PhotoZShiftandStretch object with the given tracer name."""
        raise ValueError("PhotoZShiftandStretch cannot be global.")


class SourceGalaxySelectField(
    SourceGalaxySystematic[_SourceGalaxyArgsT], Generic[_SourceGalaxyArgsT]
):
    """The source galaxy select field systematic.

    A systematic that allows specifying the 3D field that will be used
    to select the 3D power spectrum when computing the angular power
    spectrum.
    """

    def __init__(self, field: str = "delta_matter"):
        """Specify which 3D field should be used when computing angular power spectra.

        :param field: the name of the 3D field that is associated to the tracer.
        """
        super().__init__()
        self.field = field

    def apply(
        self, tools: ModelingTools, tracer_arg: _SourceGalaxyArgsT
    ) -> _SourceGalaxyArgsT:
        """Apply method to include systematics in the tracer_arg.

        :param tools: the modeling tools used to update the tracer_arg
        :param tracer_arg: the original source galaxy tracer arg to which we
            apply the systematics.
        :return: a new source galaxy tracer arg with the systematic applied
        """
        return replace(tracer_arg, field=self.field)


class SourceGalaxy(Source, Generic[_SourceGalaxyArgsT]):
    """Source class for galaxy based sources."""

    def __init__(
        self,
        *,
        sacc_tracer: str,
        systematics: None | Sequence[SourceGalaxySystematic] = None,
    ):
        """Initialize the SourceGalaxy object.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(sacc_tracer)

        self.sacc_tracer = sacc_tracer
        self.current_tracer_args: None | _SourceGalaxyArgsT = None
        self.systematics: UpdatableCollection[SourceGalaxySystematic] = (
            UpdatableCollection(systematics)
        )
        self.tracer_args: _SourceGalaxyArgsT

    def _read(self, sacc_data: sacc.Sacc) -> None:
        """Read the galaxy redshift distribution model from a sacc file.

        All derived classes must call this method in their own `_read` method
        after they have read their own data and initialized their tracer_args.

        :param sacc_data: The SACC data object to be read
        """
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
