"""Deprecated module with classes related to the Supernova statist."""

# flake8: noqa

import warnings

import firecrown.likelihood.supernova

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.supernova import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import


assert not hasattr(firecrown.likelihood.supernova, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.supernova instead.",
    DeprecationWarning,
    stacklevel=2,
)
