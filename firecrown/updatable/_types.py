"""Type definitions for the updatable module."""

from __future__ import annotations
from typing import Protocol, runtime_checkable

from firecrown.parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
    UpdatableUsageRecord,
)


@runtime_checkable
class UpdatableProtocol(Protocol):
    """Protocol defining the interface for updatable objects.

    Both Updatable and UpdatableCollection implement this interface.
    The @runtime_checkable decorator allows isinstance() checks.
    """

    _updated: bool

    def update(
        self,
        params: ParamsMap,
        updated_record: list[UpdatableUsageRecord] | None = None,
    ) -> None:
        """Update the object with new parameters."""

    def reset(self) -> None:
        """Reset the object to its initial state."""

    def is_updated(self) -> bool:
        """Check if the object has been updated."""

    def required_parameters(self) -> RequiredParameters:
        """Return the required parameters for this object."""

    def get_derived_parameters(self) -> None | DerivedParameterCollection:
        """Get the derived parameters for this object."""
