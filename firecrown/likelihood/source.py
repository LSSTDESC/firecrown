"""Abstract base classes for TwoPoint Statistics sources."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import (
    ClassVar,
    Generic,
    Sequence,
    TypeVar,
    final,
    Annotated,
    Literal,
    Protocol,
)

from pydantic import BaseModel, ConfigDict, Field
import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.nl_pt
import sacc
from scipy.interpolate import Akima1DInterpolator

from firecrown import parameters
from firecrown.likelihood.weak_lensing import SupportsWeakLensingApply
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

    cosmo_hash: None | int
    tracers: Sequence[Tracer]

    def __init__(self, sacc_tracer: str) -> None:
        """Create a Source object that uses the named tracer.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)
        self.sacc_tracer = sacc_tracer

    @abstractmethod
    def read_systematics(self, sacc_data: sacc.Sacc) -> None:
        """Abstract method to read the systematics for this source from the SACC file.

        :param sacc_data: The SACC data object to be read
        """

    @final
    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this source from the SACC file.

        :param sacc_data: The SACC data object to be read
        """
        self.read_systematics(sacc_data)
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


class Source(Protocol[UpdatableT]):
    def get_tracers(self, tools: ModelingTools) -> list[Tracer]: ...
    def get_scale(self) -> float: ...
    def read(self, sacc: Sacc) -> None: ...


# class CMBSource(Updatable):
#     def get_tracers(self):
#         pass
#     def get_scale(self):
#         pass
#     def read(self):
#         pass


class Tracer:
    """Extending the pyccl.Tracer object with additional information.

    Bundles together a pyccl.Tracer object with optional information about the
    underlying 3D field, or a pyccl.nl_pt.PTTracer and halo profiles.
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
        halo_profile: None | pyccl.halos.HaloProfile = None,
        halo_2pt: None | pyccl.halos.Profile2pt = None,
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
        self.halo_profile = halo_profile
        self.halo_2pt = halo_2pt

    @property
    def has_pt(self) -> bool:
        """Answer whether we have a perturbation theory tracer.

        :return: True if we have a pt_tracer, and False if not.
        """
        return self.pt_tracer is not None

    @property
    def has_hm(self) -> bool:
        """Answer whether we have a halo model profile.

        Return True if we have a halo_profile, and False if not.
        """
        return self.halo_profile is not None


# Sources of galaxy distributions


@dataclass(frozen=True)
class GalaxyObservableModelParameters:
    """Class for galaxy based sources arguments."""

    z: npt.NDArray[np.float64]
    dndz: npt.NDArray[np.float64]
    scale: float = 1.0
    field: str = "delta_matter"


SourceGalaxyArgs = GalaxyObservableModelParameters


class HasGalaxyObservableModel(Protocol):
    """Protocol for classes that have a galaxy observable model."""

    galaxy_observable_model: GalaxyObservableModelParameters
    __dataclass_fields__: ClassVar[dict]


# HasGalaxyModel is a constrained type variable to be used to constrain
# the type to those that implement the HasGalaxyObservableModel protocol
# (by having a galaxy_observable_model attribute).

HasGalaxyModel = TypeVar("HasGalaxyModel", bound=HasGalaxyObservableModel)


class GalaxyObservableModelCalibration:
    """Represents calibration data for model of some galaxy-based observable."""


_GalaxyObservableModelParametersT = TypeVar(
    "_GalaxyObservableModelParametersT", bound=GalaxyObservableModelParameters
)


class SourceGalaxySystematic(
    SourceSystematic, Generic[_GalaxyObservableModelParametersT]
):
    """Abstract base class for all galaxy-based source systematics."""

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: _GalaxyObservableModelParametersT
    ) -> _GalaxyObservableModelParametersT:
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


class SourceGalaxyPhotoZShift(Updatable):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some amount `delta_z`.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar delta_z: the photo-z shift.
    """

    def __init__(self, sacc_tracer: str, active: bool = True) -> None:
        """Create a PhotoZShift object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        :param active: whether to use and active or passive transformation
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.delta_z = parameters.register_new_updatable_parameter(
            default_value=SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z
        )
        if active:
            self._transform = dndz_shift_and_stretch_active
        else:
            self._transform = dndz_shift_and_stretch_passive

    def apply(self, tools: ModelingTools, tracer_arg: HasGalaxyModel) -> HasGalaxyModel:
        """Apply a shift to the photo-z distribution of a source.

        :param tools: the modeling tools use to update the tracer arg
        :param tracer_arg: the original source galaxy tracer arg to which we
            apply the systematic.
        :return: a new source galaxy tracer arg with the systematic applied
        """
        new_z, new_dndz = self._transform(
            tracer_arg.galaxy_observable_model.z,
            tracer_arg.galaxy_observable_model.dndz,
            self.delta_z,
            1.0,
        )
        galaxy_observable_model = replace(
            tracer_arg.galaxy_observable_model,
            z=new_z,
            dndz=new_dndz,
        )

        return replace(tracer_arg, galaxy_observable_model=galaxy_observable_model)


class PhotoZShift(SourceGalaxyPhotoZShift):
    """Photo-z shift systematic."""

    # TODO: Can this be deleted?


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


def dndz_shift_and_stretch_active(
    z: npt.NDArray[np.float64],
    dndz: npt.NDArray[np.float64],
    delta_z: float,
    sigma_z: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Shift and stretch the photo-z distribution using an active transformation.

    We use "makima" interpolation, a cubic spline method based on the modified Akima
    algorithm. This approach prevents overshooting when the data remains constant for
    more than two consecutive nodes. Additionally, we set `extrapolate=False` and we set
    extrapolated values to zero.

    The active transformation preserves the redshift array and modifies the dndz array.
    This transformation introduces an interpolation error on dndz.

    :param z: the redshifts
    :param dndz: the dndz
    :param delta_z: the photo-z shift
    :param sigma_z: the photo-z stretch
    :return: the shifted and stretched dndz
    """
    if sigma_z <= 0.0:
        raise ValueError("Stretch Parameter must be positive")
    # We need a small padding to avoid extrapolation at the edges
    padding = 1.0e-8
    z_padded = np.concatenate([[z[0] - padding], z, [z[-1] + padding]])
    dndz_padded = np.concatenate([[dndz[0]], dndz, [dndz[-1]]])
    dndz_interp = Akima1DInterpolator(z_padded, dndz_padded, method="makima")
    dndz_mean = np.average(z, weights=dndz)

    z_new = (z - dndz_mean + delta_z) / sigma_z + dndz_mean
    # Apply the shift and stretch
    dndz = np.nan_to_num(dndz_interp(z_new, extrapolate=False) / sigma_z)
    dndz = np.clip(dndz, 0.0, None)

    return z, dndz


def dndz_shift_and_stretch_passive(
    z: npt.NDArray[np.float64],
    dndz: npt.NDArray[np.float64],
    delta_z: float,
    sigma_z: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Shift and stretch the photo-z distribution using a passive transformation.

    :param z: the redshifts
    :param dndz: the dndz
    :param delta_z: the photo-z shift
    :param sigma_z: the photo-z stretch
    :return: the shifted and stretched dndz
    """
    if sigma_z <= 0.0:
        raise ValueError("Stretch Parameter must be positive")
    dndz_mean = np.average(z, weights=dndz)
    z_passive = sigma_z * (z - dndz_mean) - delta_z + dndz_mean
    z_passive_positive = z_passive >= 0.0
    z_new = np.atleast_1d(z_passive[z_passive_positive])
    dndz_new = np.atleast_1d(dndz[z_passive_positive] / sigma_z)

    return z_new, dndz_new


class SourceGalaxyPhotoZShiftandStretch(Updatable):
    """A photo-z shift & stretch bias.

    This systematic shifts and widens the photo-z distribution by some amount `delta_z`.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar delta_z: the photo-z shift.
    :ivar sigma_z: the photo-z stretch.
    """

    def __init__(self, sacc_tracer: str, active: bool = True) -> None:
        """Create a PhotoZShift object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        :param active: whether to use and active or passive transformation
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.delta_z = parameters.register_new_updatable_parameter(
            default_value=SOURCE_GALAXY_SYSTEMATIC_DEFAULT_DELTA_Z
        )
        self.sigma_z = parameters.register_new_updatable_parameter(
            default_value=SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z
        )

        if active:
            self._transform = dndz_shift_and_stretch_active
        else:
            self._transform = dndz_shift_and_stretch_passive

    def apply(self, _: ModelingTools, tracer_arg: HasGalaxyModel) -> HasGalaxyModel:
        """Apply a shift & stretch to the photo-z distribution of a source."""
        new_z, new_dndz = self._transform(
            tracer_arg.galaxy_observable_model.z,
            tracer_arg.galaxy_observable_model.dndz,
            self.delta_z,
            self.sigma_z,
        )
        galaxy_observable_model = replace(
            tracer_arg.galaxy_observable_model,
            z=new_z,
            dndz=new_dndz,
        )
        return replace(tracer_arg, galaxy_observable_model=galaxy_observable_model)


class PhotoZShiftandStretch(SourceGalaxyPhotoZShiftandStretch):
    """Photo-z shift and stretch systematic."""

    # TODO: Can we delete this?


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


class SourceGalaxySelectField(Updatable):
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

    def apply(self, tools: ModelingTools, tracer_arg: HasGalaxyModel) -> HasGalaxyModel:
        """Apply method to include systematics in the tracer_arg.

        :param tools: the modeling tools used to update the tracer_arg
        :param tracer_arg: the original source galaxy tracer arg to which we
            apply the systematics.
        :return: a new source galaxy tracer arg with the systematic applied
        """
        galaxy_observable_model = replace(
            tracer_arg.galaxy_observable_model,
            field=self.field,
        )
        return replace(tracer_arg, galaxy_observable_model=galaxy_observable_model)


class GalaxyModel(Updatable):
    """A model to be used in all galaxy-based sources."""

    def __init__(
        self,
        *,
        sacc_tracer: str,
        systematics: None | Sequence[HasGalaxyObservableModel] = None,
    ):
        """Initialize the GalaxyModel object.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.sacc_tracer = sacc_tracer
        self.current_tracer_args: None | GalaxyObservableModelParameters = None
        self.systematics: UpdatableCollection[SupportsWeakLensingApply] = (
            UpdatableCollection(systematics)
        )
        # TODO: change self.tracer_args to self.original_tracer_args,
        # because this is what is originally read from Sacc and never modified.
        self.tracer_args: None | GalaxyObservableModelParameters = None

    def read_systematics(self, sacc_data: sacc.Sacc) -> None:
        """Read the systematics for this source from the SACC file.

        :param sacc_data: The SACC data object to be read
        """
        for systematic in self.systematics:
            systematic.read(sacc_data)

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the galaxy redshift distribution model from a sacc file.

        All derived classes must call this method in their own `_read` method
        after they have read their own data and initialized their tracer_args.

        :param sacc_data: The SACC data object to be read
        """

        tracer = sacc_data.get_tracer(self.sacc_tracer)

        z = getattr(tracer, "z").copy().flatten()
        nz = getattr(tracer, "nz").copy().flatten()
        indices = np.argsort(z)
        z = z[indices]
        nz = nz[indices]

        self.tracer_args = GalaxyObservableModelParameters(
            z=z,
            dndz=nz,
        )


SourceGalaxy = GalaxyModel
