"""Deprecated module for number counts likelihoods."""

# flake8: noqa

import warnings

import firecrown.likelihood.number_counts

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.number_counts import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import

assert not hasattr(firecrown.likelihood.number_counts, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.number_counts instead.",
    DeprecationWarning,
    stacklevel=2,
)
