"""Deprecated module with classes related to the Student T distribution."""

# flake8: noqa

import warnings

import firecrown.likelihood.student_t

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
from firecrown.likelihood.student_t import *

# pylint: enable=unused-import,unused-wildcard-import,wildcard-import

assert not hasattr(firecrown.likelihood.student_t, "__all__")

warnings.warn(
    "This module is deprecated. Use firecrown.likelihood.student_t instead.",
    DeprecationWarning,
    stacklevel=2,
)
