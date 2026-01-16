"""DEPRECATED: Use firecrown.updatable instead.

This module provides backwards compatibility for code that imports from
firecrown.parameters. All functionality has been moved to firecrown.updatable.

This module will be removed in a future version of Firecrown.
"""

import warnings

# Re-export everything from firecrown.updatable for backward compatibility
from firecrown.updatable import (
    DerivedParameter,
    DerivedParameterCollection,
    InternalParameter,
    ParamsMap,
    RequiredParameters,
    SamplerParameter,
    handle_unused_params,
    parameter_get_full_name,
    register_new_updatable_parameter,
)

# Issue deprecation warning when this module is imported
warnings.warn(
    "The firecrown.parameters module is deprecated and will be removed in a future "
    "version. Please use firecrown.updatable instead. All parameter-related classes "
    "and functions are now available from firecrown.updatable.",
    DeprecationWarning,
    stacklevel=2,
)

# pylint: disable=duplicate-code
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
