"""Deprecated module with classes to predict binned cluster number counts."""

# flake8: noqa

import warnings

import firecrown.likelihood.binned_cluster_number_counts

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.binned_cluster_number_counts import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import

assert not hasattr(firecrown.likelihood.binned_cluster_number_counts, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.binned_cluster_number_counts instead.",
    DeprecationWarning,
    stacklevel=2,
)
