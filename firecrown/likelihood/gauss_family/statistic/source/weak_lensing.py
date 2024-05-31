"""Module for weak lensing likelihoods.
"""

import warnings

import firecrown.likelihood.weak_lensing
from firecrown.likelihood.weak_lensing import *  # pylint: disable=wildcard-import

assert not hasattr(firecrown.likelihood.weak_lensing, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.weak_lensing instead.",
    DeprecationWarning,
    stacklevel=2,
)
