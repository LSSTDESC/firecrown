"""Deprecated module for weak lensing likelihoods."""

# flake8: noqa

import warnings

import firecrown.likelihood.weak_lensing

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.weak_lensing import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import

assert not hasattr(firecrown.likelihood.weak_lensing, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.weak_lensing instead.",
    DeprecationWarning,
    stacklevel=2,
)
