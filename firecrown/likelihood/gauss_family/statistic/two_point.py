"""Deprecated module with classes related to the TwoPoint statistic."""

# flake8: noqa

import warnings

import firecrown.likelihood.two_point

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.two_point import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import

assert not hasattr(firecrown.likelihood.two_point, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.two_point instead.",
    DeprecationWarning,
    stacklevel=2,
)
