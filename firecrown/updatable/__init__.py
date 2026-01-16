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

# Import from private submodules
from ._exceptions import MissingSamplerParameterError
from ._base import Updatable
from ._records import UpdatableUsageRecord
from ._types import UpdatableProtocol
from ._collection import UpdatableCollection
from ._utils import (
    get_default_params,
    get_default_params_map,
    assert_updatable_interface,
)

# Import parameter-related classes (formerly from firecrown.parameters)
from ._parameters_derived import DerivedParameter, DerivedParameterCollection
from ._parameters_map import ParamsMap, handle_unused_params
from ._parameters_names import parameter_get_full_name
from ._parameters_required import RequiredParameters
from ._parameters_types import (
    InternalParameter,
    SamplerParameter,
    register_new_updatable_parameter,
)

__all__ = [
    # Exceptions
    "MissingSamplerParameterError",
    # Base classes and type aliases
    "Updatable",
    "UpdatableProtocol",
    # Collections
    "UpdatableCollection",
    # Utility functions
    "get_default_params",
    "get_default_params_map",
    "assert_updatable_interface",
    # Usage records
    "UpdatableUsageRecord",
    # Parameter classes (formerly from firecrown.parameters)
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
