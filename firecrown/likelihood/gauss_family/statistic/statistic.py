""""Deprecated module with classes related to Statistic."""

# flake8: noqa

import warnings

import firecrown.likelihood.statistic

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.statistic import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import


assert not hasattr(firecrown.likelihood.statistic, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.statistic instead.",
    DeprecationWarning,
    stacklevel=2,
)
