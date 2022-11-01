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
from typing import final, Dict, Optional
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


class Updatable(ABC):
    """Abstract class Updatable is the base class for Updatable objects in Firecrown.

    Any class in Firecrown that supports updating from a ParamsMap should
    inherit. Such classes are expected to change state only in through their
    implementation of _update (including any other private methods used to
    implement _update). Other functions should not change the data of Updatable
    objects.

    """

    def __init__(self):
        """Updatable initialization."""
        self._updated: bool = False
        self._returned_derived: bool = False
        self._sampler_parameters: Dict[str, SamplerParameter] = {}
        self._internal_parameters: Dict[str, InternalParameter] = {}
        self.sacc_tracer = None

    def __setattr__(self, key, value):
        if isinstance(value, SamplerParameter):
            if key in self._sampler_parameters or hasattr(self, key):
                raise ValueError(f"attribute {key} already set to the object")
            else:
                self._sampler_parameters[key] = value
                super().__setattr__(key, None)
        elif isinstance(value, InternalParameter):
            if key in self._internal_parameters or hasattr(self, key):
                raise ValueError(f"attribute {key} already set to the object")
            else:
                self._internal_parameters[key] = value
                super().__setattr__(key, value.get_value())
        else:
            super().__setattr__(key, value)

    @final
    def update(self, params: ParamsMap):
        """Update self by calling the abstract _update() method.

        :param params: new parameter values
        """
        if not self._updated:
            for parameter in self._sampler_parameters:
                value = params.get_from_prefix_param(self.sacc_tracer, parameter)
                setattr(self, parameter, value)
            self._update(params)
            self._updated = True

    @final
    def reset(self):
        """Reset self by calling the abstract _reset() method, and mark as reset."""
        self._updated = False
        self._returned_derived = False
        self._reset()

    @abstractmethod
    def _update(self, params: ParamsMap) -> None:  # pragma: no cover
        """Abstract method to be implemented by all concrete classes to update
        self.

        Concrete classes must override this, updating themselves from the given
        ParamsMap. If the supplied ParamsMap is lacking a required parameter,
        an implementation should raise a TypeError.

        The base class implementation does nothing.

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
        if not self._returned_derived:
            self._returned_derived = True
            return self._get_derived_parameters()

        return None

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
    Updatable.

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
        else:
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
