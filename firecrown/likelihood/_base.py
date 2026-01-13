"""Base classes for likelihood framework.

This module contains all abstract base classes and core types used throughout
the likelihood package. It must not import from any other private modules in
this package to avoid circular dependencies.

Classes moved from:
- _likelihood.py: Likelihood, NamedParameters
- _statistic.py: Statistic, StatisticUnreadError, GuardedStatistic, TrivialStatistic
- _source.py: Source, SourceSystematic, Tracer, SourceGalaxyArgs,
              SourceGalaxySystematic, SourceGalaxyPhotoZShift,
              SourceGalaxyPhotoZShiftandStretch, SourceGalaxySelectField, SourceGalaxy,
              PhotoZShift, PhotoZShiftFactory, PhotoZShiftandStretch,
              PhotoZShiftandStretchFactory, helper functions
"""

# pylint: disable=too-many-lines
# This module consolidates base classes to avoid circular dependencies

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Generic, Literal, TypeVar, final

import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.nl_pt
import sacc
from pydantic import BaseModel, ConfigDict, Field
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import fftconvolve

from firecrown import parameters
from firecrown.data_types import DataVector, TheoryVector
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import (
    DerivedParameterCollection,
    ParamsMap,
    RequiredParameters,
)
from firecrown.updatable import Updatable, UpdatableCollection


# ============================================================================
# Classes from _likelihood.py
# ============================================================================


class Likelihood(Updatable):
    """Likelihood is an abstract class.

    Concrete subclasses represent specific likelihood forms (e.g. gaussian with
    constant covariance matrix, or Student's t, etc.).

    Concrete subclasses must have an implementation of both :meth:`read` and
    :meth:`compute_loglike`. Note that abstract subclasses of Likelihood might implement
    these methods, and provide other abstract methods for their subclasses to implement.
    """

    def __init__(
        self,
        *,
        parameter_prefix: None | str = None,
        raise_on_unused_parameter: bool = True,
    ) -> None:
        """Default initialization for a base Likelihood object.

        :params parameter_prefix: The prefix to prepend to all parameter names
        """
        super().__init__(parameter_prefix=parameter_prefix)
        self.raise_on_unused_parameter = raise_on_unused_parameter

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrix for this likelihood from the SACC file.

        :param sacc_data: The SACC data object to be read
        """

    def make_realization_vector(self) -> npt.NDArray[np.float64]:
        """Create a new realization of the model.

        This new realization uses the previously computed theory vector and covariance
        matrix.

        :return: the new realization of the theory vector
        """
        raise NotImplementedError(
            "This class does not implement make_realization_vector."
        )

    def make_realization(
        self, sacc_data: sacc.Sacc, add_noise: bool = True, strict: bool = True
    ) -> sacc.Sacc:
        """Create a new realization of the model.

        This realization uses the previously computed theory vector and covariance
        matrix.

        :param sacc_data: The SACC data object containing the covariance matrix
        :param add_noise: If True, add noise to the realization. If False, return
            only the theory vector.
        :param strict: If True, check that the indices of the realization cover
            all the indices of the SACC data object.

        :return: the new SACC object containing the new realization
        """

    def compute_loglike_for_sampling(self, tools: ModelingTools) -> float:
        """Compute the log-likelihood of generic CCL data, swallowing some CCL errors.

         If CCL raises an error indicating an integration error, this function
         returns -np.inf.

        :param tools: the ModelingTools to be used in calculating the likelihood
        :return: the log-likelihood
        """
        try:
            return self.compute_loglike(tools)
        except pyccl.errors.CCLError as e:
            if e.args[0].startswith("Error CCL_ERROR"):
                warnings.warn(f"CCL error:\n{e}\nin likelihood, returning -inf")
                return -np.inf
            raise

    @abstractmethod
    def compute_loglike(self, tools: ModelingTools) -> float:
        """Compute the log-likelihood of generic CCL data.

        :param tools: the ModelingTools to be used in calculating the likelihood
        :return: the log-likelihood
        """


class NamedParameters:
    """Provides access to a set of parameters of a given set of types.

    Access to the parameters is provided by a type-safe interface. Each of the
    access functions assures that the parameter value it returns is of the
    specified type.

    """

    def __init__(
        self,
        mapping: (
            None
            | Mapping[
                str,
                str
                | int
                | bool
                | float
                | npt.NDArray[np.int64]
                | npt.NDArray[np.float64],
            ]
        ) = None,
    ):
        """Initialize the object from the supplied mapping of values.

        :param mapping: the mapping from strings to values used for initialization
        """
        if mapping is None:
            self.data = {}
        else:
            self.data = dict(mapping)

    def get_bool(self, name: str, default_value: None | bool = None) -> bool:
        """Return the named parameter as a bool.

        :param name: the name of the parameter to be returned
        :param default_value: the default value if the parameter is not found
        :return: the value of the parameter (or the default value)
        """
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, bool)
        return val

    def get_string(self, name: str, default_value: None | str = None) -> str:
        """Return the named parameter as a string.

        :param name: the name of the parameter to be returned
        :param default_value: the default value if the parameter is not found
        :return: the value of the parameter (or the default value)
        """
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, str)
        return val

    def get_int(self, name: str, default_value: None | int = None) -> int:
        """Return the named parameter as an int.

        :param name: the name of the parameter to be returned
        :param default_value: the default value if the parameter is not found
        :return: the value of the parameter (or the default value)
        """
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, int)
        return val

    def get_float(self, name: str, default_value: None | float = None) -> float:
        """Return the named parameter as a float.

        :param name: the name of the parameter to be returned
        :param default_value: the default value if the parameter is not found
        :return: the value of the parameter (or the default value)
        """
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, float)
        return val

    def get_int_array(self, name: str) -> npt.NDArray[np.int64]:
        """Return the named parameter as a numpy array of int.

        :param name: the name of the parameter to be returned
        :return: the value of the parameter
        """
        tmp = self.data[name]
        assert isinstance(tmp, np.ndarray)
        val = tmp.view(dtype=np.int64)
        assert val.dtype == np.int64
        return val

    def get_float_array(self, name: str) -> npt.NDArray[np.float64]:
        """Return the named parameter as a numpy array of float.

        :param name: the name of the parameter to be returned
        :return: the value of the parameter
        """
        tmp = self.data[name]
        assert isinstance(tmp, np.ndarray)
        val = tmp.view(dtype=np.float64)
        assert val.dtype == np.float64
        return val

    def to_set(
        self,
    ) -> set[
        str | int | bool | float | npt.NDArray[np.int64] | npt.NDArray[np.float64]
    ]:
        """Return the contained data as a set.

        :return: the value of the parameter as a set
        """
        return set(self.data)

    def set_from_basic_dict(
        self,
        basic_dict: dict[
            str,
            str | float | int | bool | Sequence[float] | Sequence[int] | Sequence[bool],
        ],
    ) -> None:
        """Set the contained data from a dictionary of basic types.

        :param basic_dict: the mapping from strings to values used for initialization
        """
        for key, value in basic_dict.items():
            if isinstance(value, (str, float, int, bool)):
                self.data = dict(self.data, **{key: value})
            elif isinstance(value, Sequence):
                if all(isinstance(v, float) for v in value):
                    self.data = dict(self.data, **{key: np.array(value)})
                elif all(isinstance(v, bool) for v in value) or all(
                    isinstance(v, int) for v in value
                ):
                    self.data = dict(
                        self.data, **{key: np.array(value, dtype=np.int64)}
                    )
                else:
                    raise ValueError(f"Invalid type for sequence value: {value}")
            else:
                raise ValueError(f"Invalid type for value: {value}")

    def convert_to_basic_dict(
        self,
    ) -> dict[
        str,
        str | float | int | bool | Sequence[float] | Sequence[int] | Sequence[bool],
    ]:
        """Convert a NamedParameters object to a dictionary of built-in types.

        :return: a dictionary containing the parameters as built-in Python types
        """
        basic_dict: dict[
            str,
            str | float | int | bool | Sequence[float] | Sequence[int] | Sequence[bool],
        ] = {}

        for key, value in self.data.items():
            if isinstance(value, (str, float, int, bool)):
                basic_dict[key] = value
            elif isinstance(value, np.ndarray):
                if value.dtype in (np.int64, np.float64):
                    basic_dict[key] = value.ravel().tolist()
                else:
                    raise ValueError(f"Invalid type for sequence value: {value}")
            else:
                raise ValueError(f"Invalid type for value: {value}")
        return basic_dict

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the NamedParameters object.

        :param key: the key to check
        :return: True if the key is in the NamedParameters object, False otherwise
        """
        return key in self.data


# ============================================================================
# Classes from _statistic.py
# ============================================================================


class StatisticUnreadError(RuntimeError):
    """Error raised when accessing an un-read statistic.

    Run-time error indicating an attempt has been made to use a statistic
    that has not had `read` called in it.
    """

    def __init__(self, stat: Statistic):
        """Initialize a new StatisticUnreadError.

        :param stat: the statistic that was accessed before `read` was called
        """
        msg = (
            f"The statistic {stat} was used for calculation before `read` "
            f"was called.\nIt may be that a likelihood factory function did not"
            f"call `read` before returning the likelihood."
        )
        super().__init__(msg)
        self.statistic = stat


class Statistic(Updatable):
    """The abstract base class for all physics-related statistics.

    Statistics read data from a SACC object as part of a multi-phase
    initialization. They manage a :class:`DataVector` and, given a
    :class:`ModelingTools` object, can compute a :class:`TheoryVector`.

    Statistics represent things like two-point functions and mass functions.
    """

    def __init__(self, parameter_prefix: None | str = None):
        """Initialize a new Statistic.

        Derived classes should make sure to class this method using:

        .. code-block:: python

            super().__init__(parameter_prefix=parameter_prefix)

        as the first thing they do in `__init__`.

        :param parameter_prefix: The prefix to prepend to all parameter names
        """
        super().__init__(parameter_prefix=parameter_prefix)
        self.sacc_indices: None | npt.NDArray[np.int64]
        self.ready = False
        self.computed_theory_vector = False
        self.theory_vector: None | TheoryVector = None

    def read(self, _: sacc.Sacc) -> None:
        """Read the data for this statistic and mark it as ready for use.

        Derived classes that override this function should make sure to call the
        base class method using:

        .. code-block:: python

            super().read(sacc_data)

        as the last thing they do.

        :param _: currently unused, but required by the interface.
        """
        assert len(self.get_data_vector()) > 0
        self.ready = True

    def _reset(self):
        """Reset this statistic.

        Derived classes that override this function should make sure to call the
        base class method using:

        .. code-block:: python

            super()._reset()

        as the last thing they do.
        """
        self.computed_theory_vector = False
        self.theory_vector = None

    @abstractmethod
    def get_data_vector(self) -> DataVector:
        """Gets the statistic data vector.

        :return: The data vector.
        """

    @final
    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, applying any systematics.

        :param tools: the modeling tools used to compute the theory vector.
        :return: The computed theory vector.
        """
        if not self.is_updated():
            raise RuntimeError(
                f"The statistic {self} has not been updated with parameters."
            )
        self.theory_vector = self._compute_theory_vector(tools)
        self.computed_theory_vector = True

        return self.theory_vector

    @abstractmethod
    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, concrete implementation."""

    def get_theory_vector(self) -> TheoryVector:
        """Returns the last computed theory vector.

        Raises a RuntimeError if the vector has not been computed.

        :return: The already-computed theory vector.
        """
        if not self.computed_theory_vector:
            raise RuntimeError(
                f"The theory for statistic {self} has not been computed yet."
            )
        assert self.theory_vector is not None, (
            "implementation error, "
            "computed_theory_vector is True but theory_vector is None"
        )
        return self.theory_vector


class GuardedStatistic(Updatable):
    """An internal class used to maintain state on statistics.

    :class:`GuardedStatistic` is used by the framework to maintain and
    validate the state of instances of classes derived from :class:`Statistic`.
    """

    def __init__(self, stat: Statistic):
        """Initialize the GuardedStatistic to contain the given :class:`Statistic`.

        :param stat: The statistic to wrap.
        """
        super().__init__()
        assert isinstance(stat, Statistic)
        self.statistic = stat

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read whatever data is needed from the given :class:`sacc.Sacc` object.

        After this function is called, the object should be prepared for the
        calling of the methods :meth:`get_data_vector` and
        :meth:`compute_theory_vector`.

        :param sacc_data: The SACC data object to read from.
        """
        if self.statistic.ready:
            raise RuntimeError("Firecrown has called read twice on a GuardedStatistic")
        self.statistic.read(sacc_data)

    def get_data_vector(self) -> DataVector:
        """Return the contained :class:`Statistic`'s data vector.

        :class:`GuardedStatistic` ensures that :meth:`read` has been called.
        first.

        :return: The most recently calculated  data vector.
        """
        if not self.statistic.ready:
            raise StatisticUnreadError(self.statistic)
        return self.statistic.get_data_vector()

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Return the contained :class:`Statistic`'s computed theory vector.

        :class:`GuardedStatistic` ensures that :meth:`read` has been called.
        first.

        :param tools: the modeling tools used to compute the theory vector.
        :return: The computed theory vector.
        """
        if not self.statistic.ready:
            raise StatisticUnreadError(self.statistic)
        return self.statistic.compute_theory_vector(tools)


class TrivialStatistic(Statistic):
    """A minimal statistic only to be used for testing Gaussian likelihoods.

    It returns a :class:`DataVector` and :class:`TheoryVector` each of which is
    three elements long. The SACC data provided to :meth:`TrivialStatistic.read`
    must supply the necessary values.
    """

    def __init__(self) -> None:
        """Initialize this statistic."""
        super().__init__()
        # Data and theory will both be of length self.count
        self.count = 3
        self.data_vector: None | DataVector = None
        self.mean = parameters.register_new_updatable_parameter(default_value=0.0)
        self.computed_theory_vector = False

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the necessary items from the sacc data.

        :param sacc_data: The SACC data object to be read
        """
        our_data = sacc_data.get_mean(data_type="count")
        assert len(our_data) == self.count
        self.data_vector = DataVector.from_list(our_data)
        self.sacc_indices = np.arange(len(self.data_vector))
        super().read(sacc_data)

    @final
    def _required_parameters(self) -> RequiredParameters:
        """Return an empty RequiredParameters.

        :return: an empty RequiredParameters.
        """
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Return an empty DerivedParameterCollection.

        :return: an empty DerivedParameterCollection.
        """
        return DerivedParameterCollection([])

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none.

        :return: The data vector.
        """
        assert self.data_vector is not None
        return self.data_vector

    def _compute_theory_vector(self, _: ModelingTools) -> TheoryVector:
        """Return a fixed theory vector.

        :param _: unused, but required by the interface
        :return: A fixed theory vector
        """
        return TheoryVector.from_list([self.mean] * self.count)


# ============================================================================
# Classes from _source.py
# ============================================================================


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
SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_V = 0.0


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

    def apply(
        self, tools: ModelingTools, tracer_arg: _SourceGalaxyArgsT
    ) -> _SourceGalaxyArgsT:
        """Apply a shift to the photo-z distribution of a source.

        :param tools: the modeling tools use to update the tracer arg
        :param tracer_arg: the original source galaxy tracer arg to which we
            apply the systematic.
        :return: a new source galaxy tracer arg with the systematic applied
        """
        new_z, new_dndz = self._transform(
            tracer_arg.z, tracer_arg.dndz, self.delta_z, 1.0
        )
        return replace(tracer_arg, z=new_z, dndz=new_dndz)


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

    def __init__(self, sacc_tracer: str, active: bool = True) -> None:
        """Create a PhotoZShift object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        :param active: whether to use and active or passive transformation
        """
        super().__init__(sacc_tracer)

        self.sigma_z = parameters.register_new_updatable_parameter(
            default_value=SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_Z
        )

        if active:
            self._transform = dndz_shift_and_stretch_active
        else:
            self._transform = dndz_shift_and_stretch_passive

    def apply(self, _: ModelingTools, tracer_arg: _SourceGalaxyArgsT):
        """Apply a shift & stretch to the photo-z distribution of a source."""
        new_z, new_dndz = self._transform(
            tracer_arg.z, tracer_arg.dndz, self.delta_z, self.sigma_z
        )
        return replace(tracer_arg, z=new_z, dndz=new_dndz)


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


def dndz_stretch_fog_gaussian(
    z,
    dndz,
    sigma_v: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Apply the Fingers-of-God (FoG) effect on the spec-z distribution.

    This function applies a gaussian kernel to model the FoG effect
    due to velocity dispersion within a redshift bin.

    :param z: the redshifts
    :param dndz: the dndz
    :param sigma_v: the velocity dispersion within the redshift bin in km/s
    :return: the shifted and stretched dndz
    """
    z = np.asarray(z)
    dndz = np.asarray(dndz)
    c = 299792.458  # km/s

    if sigma_v < 0.0:
        raise ValueError("Stretch Parameter (sigma_v) must be positive")
    if sigma_v == 0.0:
        return z, dndz

    n = z.shape[0]
    dz = z[1] - z[0]

    z_mean = np.trapezoid(dndz * z, z)
    sigma_s = (1 + z_mean) * sigma_v / c

    z_kernel = (np.arange(n) - n // 2) * dz
    kernel = np.exp(-0.5 * (z_kernel / sigma_s) ** 2)
    kernel /= np.trapezoid(kernel, z)

    dndz_new = fftconvolve(dndz, kernel, mode="same")
    dndz_new /= np.trapezoid(dndz_new, z)

    return z, dndz_new


def dndz_stretch_fog_lorentzian(
    z,
    dndz,
    sigma_v: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Apply the Fingers-of-God (FoG) effect on the spec-z distribution.

    This function applies a lorentzian kernel to model the FoG effect
    due to velocity dispersion within a redshift bin.

    :param z: the redshifts
    :param dndz: the dndz
    :param sigma_v: the velocity dispersion within the redshift bin in km/s
    :return: the shifted and stretched dndz
    """
    z = np.asarray(z)
    dndz = np.asarray(dndz)
    c = 299792.458  # km/s

    if sigma_v < 0.0:
        raise ValueError("Stretch Parameter (sigma_v) must be positive")
    if sigma_v == 0.0:
        return z, dndz

    n = z.shape[0]
    dz = z[1] - z[0]

    z_mean = np.trapezoid(dndz * z, z)
    sigma_s = (1 + z_mean) * sigma_v / c

    z_kernel = (np.arange(n) - n // 2) * dz
    arg = -np.sqrt(2) * np.abs(z_kernel) / sigma_s
    arg = np.clip(arg, -700, 0)  # avoid overflow
    kernel = np.exp(arg)
    kernel /= np.trapezoid(kernel, z)

    dndz_new = fftconvolve(dndz, kernel, mode="same")
    dndz_new /= np.trapezoid(dndz_new, z)

    return z, dndz_new


class SourceGalaxySpecZStretch(
    SourceGalaxySystematic[_SourceGalaxyArgsT], Generic[_SourceGalaxyArgsT]
):
    """A spec-z convolution for the Fingers-of-God (FoG) effect.

    This systematic convolves the spec-z distribution with a kernel to model
    the FoG effect.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar sigma_v: the spec-z stretch.
    """

    def __init__(self, sacc_tracer: str, kernel: str = "gaussian") -> None:
        """Create a SpecZShift object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        :param kernel: which kernel to use when convolving the distribution.
        """
        super().__init__(sacc_tracer)

        self.sigma_v = parameters.register_new_updatable_parameter(
            default_value=SOURCE_GALAXY_SYSTEMATIC_DEFAULT_SIGMA_V
        )

        if kernel == "lorentzian":
            self._transform = dndz_stretch_fog_lorentzian
        else:
            self._transform = dndz_stretch_fog_gaussian

    def apply(self, _: ModelingTools, tracer_arg: _SourceGalaxyArgsT):
        """Apply a convolution to the spec-z distribution of a source."""
        new_z, new_dndz = self._transform(tracer_arg.z, tracer_arg.dndz, self.sigma_v)
        return replace(tracer_arg, z=new_z, dndz=new_dndz)


class SpecZStretch(SourceGalaxySpecZStretch):
    """Spec-z stretch systematic."""


class SpecZStretchFactory(BaseModel):
    """Factory class for SpecZStretch objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["SpecZStretchFactory"],
        Field(description="The type of the systematic."),
    ] = "SpecZStretchFactory"

    def create(self, bin_name: str) -> SpecZStretch:
        """Create a SpecZStretch object with the given tracer name."""
        return SpecZStretch(bin_name)

    def create_global(self) -> SpecZStretch:
        """Create a SpecZStretch object with the given tracer name."""
        raise ValueError("SpecZStretch cannot be global.")


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

    def read_systematics(self, sacc_data: sacc.Sacc) -> None:
        """Read the systematics for this source from the SACC file.

        :param sacc_data: The SACC data object to be read
        """
        for systematic in self.systematics:
            systematic.read(sacc_data)

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

        z = tracer.z.copy().flatten()
        nz = tracer.nz.copy().flatten()
        indices = np.argsort(z)
        z = z[indices]
        nz = nz[indices]

        self.tracer_args = replace(
            tracer_args,
            z=z,
            dndz=nz,
        )
