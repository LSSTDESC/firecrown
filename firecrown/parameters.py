"""


Parameters module
=================
Provide classes and functions to support groups of named parameters.

Classes include:
    ParamsMap: specialized Dict mapping strings to floats.
    RequiredParameters: represents a collection of required parameter names.

"""

from __future__ import annotations
from typing import List, Dict, Optional


def parameter_get_full_name(prefix: Optional[str], param: str) -> str:
    """Form a full parameter name from the given (optional) prefix and name.

    Parameter names, as stored in SACC, for example, contain an optional
    prefix; if a prefix is present, it will be separated from the name by
    an underscore.

    Prefixes and names should avoid containing embedded underscores. This
    is currently not enforced in the code.

    The parameter name can not be empty, even if accompanied by a prefix;
    this is enforced in the code.

    Ill-formed parameter names result in raising a ValueError.
    """
    if len(param) == 0:
        raise ValueError("param must not be an empty string")

    if prefix:
        return f"{prefix}_{param}"
    return param


class ParamsMap(Dict[str, float]):
    """A specialized Dict in which all keys are strings and values are floats.

    The recommended access method is get_from_prefix_param, rather than indexing
    with square brackets like x[].
    """

    def get_from_prefix_param(self, prefix: Optional[str], param: str) -> float:
        """Return the parameter identified by the optional prefix and parameter name.


        See parameter_get_full_name for rules on the forming of prefix and name.
        Raises a KeyError if the parameter is not found.
        """
        fullname = parameter_get_full_name(prefix, param)

        if fullname in self.keys():
            return self[fullname]
        raise KeyError(f"Prefix `{prefix}`, key `{param}' not found.")


class RequiredParameters:
    """Represents a sequence of parameter names.

    This class provides some type safety to distinguish between an arbitrary
    list of strings, and one intended to be a list of required parameter names.

    An instance can be created from a list of strings.
    Instances can be concatenated using `+`, and compared for equality using `==`.

    To iterate through the names (which are strings), use `get+params_names`,
    which implements lazy evaluation.
    """

    def __init__(self, params_names: List[str]):
        """Construct an instance from a list of strings."""
        self.params_names = params_names

    def __add__(self, other: RequiredParameters):
        """Return a new RequiredParameters with the concatenated names.

        Note that this function returns a new object that does not share state
        with either argument to the addition operator."""
        return RequiredParameters(self.params_names + other.params_names)

    def __eq__(self, other: object):
        """Compare two RequiredParameters objects for equality.

        This implementation raises a NotImplemented exception unless both
        objects are RequireParameters objects.

        Two RequireParameters objects are equal if their contained names
        are equal (including appearing in the same order).
        """
        if not isinstance(other, RequiredParameters):
            return NotImplemented
        return self.params_names == other.params_names

    def get_params_names(self):
        """Implement lazy iteration through the contained parameter names."""
        for name in self.params_names:
            yield name
