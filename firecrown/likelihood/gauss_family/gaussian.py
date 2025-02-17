"""Deprecated module with classes related to the Gaussian likelihood."""

# flake8: noqa

import warnings

import firecrown.likelihood.gaussian

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.gaussian import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import

assert not hasattr(firecrown.likelihood.gaussian, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.gaussian instead.",
    DeprecationWarning,
    stacklevel=2,
)
