"""Parameters that can be updated, and collections of them.

Abstract class :class:`Updatable` is the base class from which any class in Firecrown
that supports updating from a :class:`ParamsMap` should inherit. Such classes are
expected to change state only in through their implementation of :meth:`_update`
(including any other private methods used to implement :meth:`_update`). Other
functions should not change the data of :class:`Updatable` objects.

:class:`UpdatableCollection` is a subclass of the built-in list. It implements the
:class:`Updatable` interface by calling :meth:`update()` on each element it contains.
The :meth:`append()` method is overridden to make sure that only objects which are of
a type that implements :class:`Updatable` can be appended to the list.

"""

from __future__ import annotations

from abc import ABC
from collections import UserList
from typing import Any, Generic, Iterable, TypeAlias, TypeVar, Union, cast, final

from firecrown.parameters import (
    InternalParameter,
    ParamsMap,
    RequiredParameters,
    SamplerParameter,
)

from .parameters import DerivedParameterCollection

GeneralUpdatable: TypeAlias = Union["Updatable", "UpdatableCollection"]


class MissingSamplerParameterError(RuntimeError):
    """Error for when a required parameter is missing.

    Raised when an Updatable fails to be updated because the ParamsMap supplied for the
    update is missing a parameter that should have been provided by the sampler.
    """

    def __init__(self, parameter: str) -> None:
        """Create the error, with a meaningful error message.

        :param parameter: name of the missing parameter
        """
        self.parameter = parameter
        msg = (
            f"The parameter `{parameter}` is required to update "
            f"something in this likelihood.\nIt should have been supplied "
            f"by the sampling framework.\nThe object being updated was:\n"
            f"{self}\n"
        )
        super().__init__(msg)


class Updatable(ABC):
    """Abstract class Updatable is the base class for Updatable objects in Firecrown.

    Any class in Firecrown that supports updating from a ParamsMap should
    inherit from :class:`Updatable`. Such classes are expected to change state
    only in through their implementation of _update (including any other private
    methods used to implement _update). Other functions should not change the
    data of Updatable objects.
    """

    def __init__(self, parameter_prefix: None | str = None) -> None:
        """Updatable initialization.

        Parameters created by firecrown.parameters.create will have a prefix
        that is given by the prefix argument to the Updatable constructor. This
        prefix is used to create the full name of the parameter. If `parameter_prefix`
        is None, then the parameter will have no prefix.

        :param parameter_prefix: prefix for all parameters in this Updatable
        """
        self._updated: bool = False
        self._returned_derived: bool = False
        self._sampler_parameters: list[SamplerParameter] = []
        self._internal_parameters: dict[str, InternalParameter] = {}
        self._updatables: list[GeneralUpdatable] = []
        self.parameter_prefix: None | str = parameter_prefix

    def __setattr__(self, key: str, value: Any) -> None:
        """Set the attribute named :attr:`key` to the supplied `value`.

        There is special handling for two types: :class:`SamplerParameter`
        and :class:`InternalParameter`.

        We also keep track of all :class:`Updatable` instance variables added,
        appending a reference to each to :attr:`self._updatables` as well as
        storing the attribute directly.

        :param key: name of the attribute
        :param value: value for the attribute
        """
        if isinstance(value, (Updatable, UpdatableCollection)):
            self._updatables.append(value)
        elif isinstance(value, Iterable):
            # Consider making this a recursive call that handles nested
            # iterables.
            for v in value:
                if isinstance(v, (Updatable, UpdatableCollection)):
                    self._updatables.append(v)

        if isinstance(value, (InternalParameter, SamplerParameter)):
            self.set_parameter(key, value)
        else:
            super().__setattr__(key, value)

    def set_parameter(
        self, key: str, value: InternalParameter | SamplerParameter
    ) -> None:
        """Sets the parameter to the given value.

        Assure this InternalParameter or SamplerParameter has not already
        been set, and then set it.

        :param key: name of the attribute
        :param value: value for the attribute
        """
        if isinstance(value, SamplerParameter):
            value.set_fullname(self.parameter_prefix, key)
            self.set_sampler_parameter(value)
        elif isinstance(value, InternalParameter):
            self.set_internal_parameter(key, value)

    def set_internal_parameter(self, key: str, value: InternalParameter) -> None:
        """Assure this InternalParameter has not already been set, and then set it.

        :param key: name of the attribute
        :param value: value for the attribute
        """
        if not isinstance(value, InternalParameter):
            raise TypeError(
                "Can only add InternalParameter objects to internal_parameters"
            )

        if key in self._internal_parameters or hasattr(self, key):
            raise ValueError(
                f"attribute {key} already set in {self} "
                f"from a parameter supplied in the likelihood factory code"
            )
        self._internal_parameters[key] = value
        super().__setattr__(key, value.get_value())

    def set_sampler_parameter(self, value: SamplerParameter) -> None:
        """Assure this SamplerParameter has not already been set, and then set it.

        :param value: value for the attribute
        """
        if not isinstance(value, SamplerParameter):
            raise TypeError(
                "Can only add SamplerParameter objects to sampler_parameters"
            )

        if value in self._sampler_parameters or hasattr(self, value.name):
            raise ValueError(
                f"attribute {value.name} already set in {self} "
                f"from a parameter read from the sampler"
            )
        self._sampler_parameters.append(value)
        super().__setattr__(value.name, None)

    @final
    def update(self, params: ParamsMap) -> None:
        """Update self by calling to prepare for the next MCMC sample.

        We first update the values of sampler parameters from the values in
        `params`. An error will be raised if any of self's sampler
        parameters can not be found in `params` or if any internal
        parameters are provided in `params`.

        We then use the `params` to update each contained :class:`Updatable` or
        :class:`UpdatableCollection` object. The method :meth:`_update` is
        called to give subclasses an opportunity to do any other preparation
        for the next MCMC sample.

        :param params: new parameter values
        """
        if self._updated:
            return

        internal_params = self._internal_parameters.keys() & params.keys()
        if internal_params:
            raise TypeError(
                f"Items of type InternalParameter cannot be modified through "
                f"update, but {','.join(internal_params)} was specified."
            )

        for parameter in self._sampler_parameters:
            try:
                value = params.get_from_full_name(parameter.fullname)
            except KeyError as exc:
                raise MissingSamplerParameterError(parameter.fullname) from exc
            setattr(self, parameter.name, value)

        for item in self._updatables:
            item.update(params)

        self._update(params)
        # We mark self as updated only after all the internal updates have
        # worked.
        self._updated = True

    def is_updated(self) -> bool:
        """Determine if the object has been updated.

        A default-constructed Updatable has not been updated. After `update`,
        but before `reset`, has been called the object is updated. After
        `reset` has been called, the object is not currently updated.

        :return:  True if the object is currently updated, and False if not.
        """
        return self._updated

    @final
    def reset(self) -> None:
        """Reset the updatable.

        Clean up self by clearing the _updated status and reseting all
        internals. We call the abstract method _reset to allow derived classes
        to clean up any additional internals.

        Each MCMC framework connector should call this after handling an MCMC
        sample.
        """
        # If we have not been updated, there is nothing to do.
        if not self._updated:
            return

        # We reset in the inverse order, first the contained updatables, then
        # the current object.
        for item in self._updatables:
            item.reset()

        # Reset the sampler parameters to None.
        for parameter in self._sampler_parameters:
            setattr(self, parameter.name, None)

        self._updated = False
        self._returned_derived = False
        self._reset()

    def _update(self, params: ParamsMap) -> None:
        """Method for auxiliary updates to be made to an updatable.

        Do any updating other than calling :meth:`update` on contained
        :class:`Updatable` objects. Implement this method in a subclass only when it
        has something to do. If the supplied :class:`ParamsMap` is lacking a required
        parameter, an implementation should raise a `TypeError`.

        This default implementation does nothing.

        :param params: a new set of parameter values
        """

    def _reset(self) -> None:  # pragma: no cover
        """Abstract method implemented by all concrete classes to update self.

        Concrete classes must override this, resetting themselves.

        The base class implementation does nothing.
        """

    @final
    def required_parameters(self) -> RequiredParameters:  # pragma: no cover
        """Returns all information about parameters required by this object.

        This object returned contains the information for all parameters
        defined in the implementing class, and any additional parameters.

        :return: a RequiredParameters object containing all relevant parameters
        """
        sampler_parameters = RequiredParameters(self._sampler_parameters)
        additional_parameters = self._required_parameters()

        for item in self._updatables:
            additional_parameters = additional_parameters + item.required_parameters()

        return sampler_parameters + additional_parameters

    def _required_parameters(self) -> RequiredParameters:  # pragma: no cover
        """Return a RequiredParameters object containing the information for this class.

        This method can be overridden by subclasses to add
        additional parameters. The default implementation returns an empty
        RequiredParameters object. This is only implemented to allow

        The base class implementation returns a list with all SamplerParameter
        objects properties.

        :return: a RequiredParameters containing all relevant parameters
        """
        return RequiredParameters([])

    @final
    def get_derived_parameters(
        self,
    ) -> None | DerivedParameterCollection:
        """Returns a collection of derived parameters.

        This occurs once per iteration of the statistical analysis. First call returns
        the DerivedParameterCollection, further calls return None.

        :return: a collection of derived parameters, or None
        """
        if not self._updated:
            raise RuntimeError(
                "Derived parameters can only be obtained after update has been called."
            )

        if self._returned_derived:
            return None

        self._returned_derived = True
        derived_parameters = self._get_derived_parameters()

        for item in self._updatables:
            derived_parameters = derived_parameters + item.get_derived_parameters()

        return derived_parameters

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Returns the derived parameters of an implementation.

        Derived classes can override this, returning a DerivedParameterCollection
        containing the derived parameters for the class. The default implementation
        returns an empty DerivedParameterCollection.

        :return: a collection of derived parameters
        """
        return DerivedParameterCollection([])


T = TypeVar("T", bound=Updatable)


class UpdatableCollection(UserList[T], Generic[T]):
    """Class that represents a collection of updatable objects.

    UpdatableCollection is a list of Updatable objects and is itself
    supports :meth:`update` and :meth:`reset` (although it does not inherit
    from :class:`Updatable`).

    Every item in an UpdatableCollection must itself be :class:`Updatable`. Calling
    :meth:`update` on the collection results in every item in the collection being
    updated.
    """

    def __init__(self, iterable: None | Iterable[T] = None) -> None:
        """Initialize the UpdatableCollection from the supplied iterable.

        If the iterable contains any object that is not Updatable, a TypeError
        is raised.

        :param iterable: An iterable that yields Updatable objects
        """
        super().__init__(iterable)
        self._updated: bool = False
        for item in self:
            if not isinstance(item, (Updatable | UpdatableCollection)):
                raise TypeError(
                    f"All the items in an UpdatableCollection must be updatable {item}"
                )

    @final
    def update(self, params: ParamsMap) -> None:
        """Update self by calling update() on each contained item.

        :param params: new parameter values for each contained item
        """
        if self._updated:
            return

        for updatable in self:
            updatable.update(params)

        self._updated = True

    def is_updated(self) -> bool:
        """Returns whether this updatable has been updated.

        Return True if the object is currently updated, and False if not.
        A default-constructed Updatable has not been updated. After `update`,
        but before `reset`, has been called the object is updated. After
        `reset` has been called, the object is not currently updated.
        """
        return self._updated

    @final
    def reset(self) -> None:
        """Resets self by calling reset() on each contained item."""
        self._updated = False
        for updatable in self:
            updatable.reset()

    @final
    def required_parameters(self) -> RequiredParameters:
        """Return a RequiredParameters object.

        The RequiredParameters object is formed by concatenating the
        RequiredParameters of each contained item.
        """
        result = RequiredParameters([])
        for updatable in self:
            result = result + updatable.required_parameters()

        return result

    @final
    def get_derived_parameters(self) -> None | DerivedParameterCollection:
        """Get all derived parameters if any."""
        has_any_derived = False
        derived_parameters = DerivedParameterCollection([])
        for updatable in self:
            derived_parameters0 = updatable.get_derived_parameters()
            if derived_parameters0 is not None:
                derived_parameters = derived_parameters + derived_parameters0
                has_any_derived = True
        if has_any_derived:
            return derived_parameters
        return None

    def append(self, item: T) -> None:
        """Append the given item to self.

        If the item is not Updatable a TypeError is raised.

        :param item: new item to be appended to the list
        """
        if not isinstance(item, Updatable):
            raise TypeError(
                "Only updatable items can be appended to an UpdatableCollection"
            )
        super().append(item)

    def __setitem__(self, key, value):
        """Set self[key] to value; raise TypeError if Value is not Updatable."""
        if not isinstance(value, Updatable):
            raise TypeError(
                "Only updatable items can be appended to an UpdatableCollection"
            )

        super().__setitem__(key, cast(T, value))


def get_default_params(*args: Updatable) -> dict[str, float]:
    """Get a ParamsMap with the default values of all parameters in the updatables.

    :param args: updatables to get the default parameters from
    :return: a ParamsMap with the default values of all parameters
    """
    updatable_collection = UpdatableCollection(args)
    required_parameters = updatable_collection.required_parameters()
    default_parameters = required_parameters.get_default_values()

    return default_parameters


def get_default_params_map(*args: Updatable) -> ParamsMap:
    """Get a ParamsMap with the default values of all parameters in the updatables.

    :param args: updatables to get the default parameters from
    :return: a ParamsMap with the default values of all parameters
    """
    return ParamsMap(get_default_params(*args))
