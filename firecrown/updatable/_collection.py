"""Collection class for updatable objects."""

from __future__ import annotations
from collections import UserList
from collections.abc import Iterable
from typing import Generic, TypeVar, cast, final

from firecrown.parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
    UpdatableUsageRecord,
)

from firecrown.updatable._base import Updatable
from firecrown.updatable._types import UpdatableProtocol


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
            if not isinstance(item, UpdatableProtocol):
                raise TypeError(
                    f"All the items in an UpdatableCollection must be updatable {item}"
                )

    @final
    def update(
        self,
        params: ParamsMap,
        updated_record: list[UpdatableUsageRecord] | None = None,
    ) -> None:
        """Update self by calling update() on each contained item.

        :param params: new parameter values for each contained item
        """
        if self._updated:
            return

        for updatable in self:
            updatable.update(params, updated_record)

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

        super().__setitem__(key, cast("T", value))
