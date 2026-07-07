"""DEPRECATED: Use firecrown.likelihood instead.

This module provides backwards compatibility for code that imports from
firecrown.likelihood.two_point. All functionality has been moved to
firecrown.likelihood._two_point and is re-exported from firecrown.likelihood.

This module will be removed in a future version of Firecrown.
"""

import warnings

# Re-export everything from firecrown.likelihood._two_point for backward
# compatibility
from firecrown.likelihood._two_point import (
    TwoPoint,
    TwoPointFactory,
    calculate_angular_cl,
    read_ell_cells,
    read_reals,
    use_source_factory,
    use_source_factory_metadata_index,
)

# Issue deprecation warning when this module is imported
warnings.warn(
    "The firecrown.likelihood.two_point module is deprecated and will be removed in "
    "a future version. Please use firecrown.likelihood instead. All two-point classes "
    "and functions are now available from firecrown.likelihood.",
    DeprecationWarning,
    stacklevel=2,
)

# pylint: disable=duplicate-code
__all__ = [
    "TwoPoint",
    "TwoPointFactory",
    "calculate_angular_cl",
    "read_ell_cells",
    "read_reals",
    "use_source_factory",
    "use_source_factory_metadata_index",
]
