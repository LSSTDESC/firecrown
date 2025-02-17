"""Deprecated module for two-point statistic sources."""

# flake8: noqa

import warnings

import firecrown.likelihood.source

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.source import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import


assert not hasattr(firecrown.likelihood.source, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.source instead.",
    DeprecationWarning,
    stacklevel=2,
)
