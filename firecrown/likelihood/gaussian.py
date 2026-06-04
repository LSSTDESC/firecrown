"""Backward compatibility shim for firecrown.likelihood.gaussian.

This module re-exports :class:`ConstGaussian` from its current location
(:mod:`firecrown.likelihood._gaussian`) to preserve compatibility with code
that imports from the old ``firecrown.likelihood.gaussian`` path.
"""

import warnings

from firecrown.likelihood._gaussian import ConstGaussian

warnings.warn(
    "firecrown.likelihood.gaussian is deprecated. "
    "Import ConstGaussian from firecrown.likelihood instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ConstGaussian"]
