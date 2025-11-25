"""Classes and functions to support groups of named parameters.

These are used in Firecrown in preference to the Python dictionary in order to
provide better type safety.
"""

from ._derived import DerivedParameter, DerivedParameterCollection
from ._map import ParamsMap, handle_unused_params
from ._names import parameter_get_full_name
from ._required import RequiredParameters
from ._types import (
    InternalParameter,
    SamplerParameter,
    register_new_updatable_parameter,
)

__all__ = [
    "DerivedParameter",
    "DerivedParameterCollection",
    "InternalParameter",
    "ParamsMap",
    "RequiredParameters",
    "SamplerParameter",
    "handle_unused_params",
    "parameter_get_full_name",
    "register_new_updatable_parameter",
]
