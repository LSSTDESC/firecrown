"""Deprecated module with classes related to the Gaussian family of statistics."""

# flake8: noqa

import warnings

import firecrown.likelihood.gaussfamily

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.gaussfamily import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import

assert not hasattr(firecrown.likelihood.gaussfamily, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.gaussfamily instead.",
    DeprecationWarning,
    stacklevel=2,
)
