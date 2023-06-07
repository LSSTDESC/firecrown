"""Parameters that can be updated, and collections of them.

Abstract class :class:`Updatable` is the base class from which any class in Firecrown
that supports updating from a :class:`ParamsMap` should inherit. Such classes are
expected to change state only in through their implementation of :python:`_update`
(including any other private methods used to implement :python:`_update`). Other
functions should not change the data of :class:`Updatable` objects.

:class:`UpdatableCollection` is a subclass of the built-in list. It implements the
:class:`Updatable` interface by calling :python:`update()` on each element it contains.
The :python:`append()` member is overridden to make sure that only objects which are of
a type that implements :class:`Updatable` can be appended to the list.

"""

from __future__ import annotations
from typing import final, Dict, Optional, Any, List, Union
from abc import ABC, abstractmethod
from collections import UserList
from .parameters import (
    ParamsMap,
    RequiredParameters,
    SamplerParameter,
    InternalParameter,
    parameter_get_full_name,
)
from .parameters import DerivedParameterCollection

GeneralUpdatable = Union["Updatable", "UpdatableCollection"]


class Updatable(ABC):
    """Abstract class Updatable is the base class for Updatable objects in Firecrown.

    Any class in Firecrown that supports updating from a ParamsMap should
    inherit. Such classes are expected to change state only in through their
    implementation of _update (including any other private methods used to
    implement _update). Other functions should not change the data of Updatable
    objects.

    """

    def __init__(self) -> None:
        """Updatable initialization."""
        self._updated: bool = False
        self._returned_derived: bool = False
        self._sampler_parameters: Dict[str, SamplerParameter] = {}
        self._internal_parameters: Dict[str, InternalParameter] = {}
        self.sacc_tracer: Optional[str] = None
        self._updatables: List[GeneralUpdatable] = []

    def __setattr__(self, key: str, value: Any) -> None:
        """Set the attribute named :python:`key` to the supplied :python:`value`.

        There is special handling for two types: :python:`SamplerParameter`
        and :python:`InternalParameter`.

        We also keep track of all :python:`Updatable` instance variables added,
        appending a reference to each to :python:`self._updatables` as well as
        storing the attribute directly.
        """
        if isinstance(value, (Updatable, UpdatableCollection)):
            self._updatables.append(value)
        if isinstance(value, SamplerParameter):
            self.set_sampler_parameter(key, value)
        elif isinstance(value, InternalParameter):
            self.set_internal_parameter(key, value)
        else:
            super().__setattr__(key, value)

    def set_internal_parameter(self, key: str, value: InternalParameter) -> None:
        """Assure this InternalParameter has not already been set, and then set it."""
        if key in self._internal_parameters or hasattr(self, key):
            raise ValueError(
                f"attribute {key} already set in {self} "
                f"from a parameter supplied in the likelihood factory code"
            )
        self._internal_parameters[key] = value
        super().__setattr__(key, value.get_value())

    def set_sampler_parameter(self, key: str, value: SamplerParameter) -> None:
        """Assure this SamplerParameter has not already been set, and then set it."""
        if key in self._sampler_parameters or hasattr(self, key):
            raise ValueError(
                f"attribute {key} already set in {self} "
                f"from a parameter read from the sampler"
            )
        self._sampler_parameters[key] = value
        super().__setattr__(key, None)

    @final
    def update(self, params: ParamsMap) -> None:
        """Update self by calling to prepare for the next MCMC sample.

        We first update the values of sampler parameters from the values in
        :python:`params`. An error will be raised if any of self's sampler
        parameters can not be found in :python:`params`.

        We then use the :python:`params` to update each contained Updatable or
        UpdatableCollection object. The method _update is called to give
        subclasses an opportunity to do any other preparation for the next
        MCMC sample.

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
                value = params.get_from_prefix_param(self.sacc_tracer, parameter)
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing required parameter "
                    f"`{parameter_get_full_name(self.sacc_tracer, parameter)}`,"
                    f" the sampling framework should provide this parameter."
                    f" The object requiring this parameter is {self}."
                ) from exc
            setattr(self, parameter, value)

        for item in self._updatables:
            item.update(params)

        self._update(params)
        # We mark self as updated only after all the internal updates have
        # worked.
        self._updated = True

    @final
    def reset(self) -> None:
        """Clean up self by clearing the _updated status and reseting all
        internals. We call the abstract method _reset to allow derived classes
        to clean up any additional internals.

        Each MCMC framework connector should call this after handling an MCMC
        sample."""
        self._updated = False
        self._returned_derived = False
        self._reset()

    def _update(self, params: ParamsMap) -> None:
        """Do any updating other than calling :python:`update` on contained
        :python:`Updatable` objects.

        Implement this method in a subclass only when it has something to do.
        If the supplied ParamsMap is lacking a required parameter,
        an implementation should raise a TypeError.

        This default implementation does nothing.

        :param params: a new set of parameter values
        """

    @abstractmethod
    def _reset(self) -> None:  # pragma: no cover
        """Abstract method to be implemented by all concrete classes to update
        self.

        Concrete classes must override this, resetting themselves.

        The base class implementation does nothing.
        """

    @final
    def required_parameters(self) -> RequiredParameters:  # pragma: no cover
        """Return a RequiredParameters object containing the information for
        all parameters defined in the implementing class, any additional
        parameter
        """

        sampler_parameters = RequiredParameters(
            [
                parameter_get_full_name(self.sacc_tracer, parameter)
                for parameter in self._sampler_parameters
            ]
        )
        additional_parameters = self._required_parameters()

        return sampler_parameters + additional_parameters

    @abstractmethod
    def _required_parameters(self) -> RequiredParameters:  # pragma: no cover
        """Return a RequiredParameters object containing the information for
        this Updatable. This method must be overridden by concrete classes.

        The base class implementation returns a list with all SamplerParameter
        objects properties.
        """

    @final
    def get_derived_parameters(
        self,
    ) -> Optional[DerivedParameterCollection]:
        """Returns a collection of derived parameters once per iteration of the
        statistical analysis. First call returns the DerivedParameterCollection,
        further calls return None.
        """
        if self._returned_derived:
            return None

        self._returned_derived = True
        return self._get_derived_parameters()

    @abstractmethod
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Abstract method to be implemented by all concrete classes to return their
        derived parameters.

        Concrete classes must override this. If no derived parameters are required
        derived classes must simply return super()._get_derived_parameters().
        """
        return DerivedParameterCollection([])


class UpdatableCollection(UserList):

    """UpdatableCollection is a list of Updatable objects and is itself
    supports :python:`update` and :python:`reset` (although it does not inherit
    from
    :python:`Updatable`).

    Every item in an UpdatableCollection must itself be Updatable. Calling
    update on the collection results in every item in the collection being
    updated.
    """

    def __init__(self, iterable=None):
        """Initialize the UpdatableCollection from the supplied iterable.

        If the iterable contains any object that is not Updatable, a TypeError
        is raised.

        :param iterable: An iterable that yields Updatable objects
        """
        super().__init__(iterable)
        for item in self:
            if not isinstance(item, Updatable):
                raise TypeError(
                    "All the items in an UpdatableCollection must be updatable"
                )

    @final
    def update(self, params: ParamsMap):
        """Update self by calling update() on each contained item.

        :param params: new parameter values
        """
        for updatable in self:
            updatable.update(params)

    @final
    def reset(self):
        """Resets self by calling reset() on each contained item."""
        for updatable in self:
            updatable.reset()

    @final
    def required_parameters(self) -> RequiredParameters:
        """Return a RequiredParameters object formed by concatenating the
        RequiredParameters of each contained item.
        """
        result = RequiredParameters([])
        for updatable in self:
            result = result + updatable.required_parameters()

        return result

    @final
    def get_derived_parameters(self) -> Optional[DerivedParameterCollection]:
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

    def append(self, item: Updatable) -> None:
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
                "Values inserted into an UpdatableCollection must be Updatable"
            )
        super().__setitem__(key, value)
