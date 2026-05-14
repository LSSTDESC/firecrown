"""Required parameters functionality."""

from __future__ import annotations
from collections.abc import Iterable, Iterator

from ._parameters_types import SamplerParameter


class RequiredParameters:
    """Represents a sequence of parameter names.

    This class provides some type safety to distinguish between an arbitrary
    list of strings, and one intended to be a list of required parameter names.

    An instance can be created from a list of strings.
    Instances can be concatenated using `+`, and compared for equality using `==`.

    To iterate through the names (which are strings), use `get+params_names`,
    which implements lazy evaluation.
    """

    def __init__(self, params: Iterable["SamplerParameter"]):
        """Construct an instance from an Iterable yielding strings."""
        self.params_set: set["SamplerParameter"] = set(params)

    def __len__(self):
        """Return the number of parameters contained."""
        return len(self.params_set)

    def __add__(self, other: RequiredParameters) -> RequiredParameters:
        """Return a new RequiredParameters with the concatenated names.

        Note that this function returns a new object that does not share state
        with either argument to the addition operator.
        """
        return RequiredParameters(self.params_set | other.params_set)

    def __sub__(self, other: RequiredParameters) -> RequiredParameters:
        """Return a new RequiredParameters with the names in self but not in other.

        Note that this function returns a new object that does not share state
        with either argument to the subtraction operator.
        """
        return RequiredParameters(self.params_set - other.params_set)

    def __eq__(self, other: object):
        """Compare two RequiredParameters objects for equality.

        This implementation raises a NotImplemented exception unless both
        objects are RequireParameters objects.

        Two RequireParameters objects are equal if their contained names
        are equal (including appearing in the same order).
        """
        if not isinstance(other, RequiredParameters):
            n = type(other).__name__
            raise TypeError(
                f"Cannot compare a RequiredParameter to an object of type {n}"
            )
        return self.params_set == other.params_set

    def get_params_names(self) -> Iterator[str]:
        """Implement lazy iteration through the contained parameter names."""
        params_names_set = set(parameter.fullname for parameter in self.params_set)
        yield from params_names_set

    def get_default_values(self) -> dict[str, float]:
        """Return a dictionary with the default values of the parameters."""
        default_values = {}
        for parameter in self.params_set:
            default_values[parameter.fullname] = parameter.get_default_value()

        return default_values
