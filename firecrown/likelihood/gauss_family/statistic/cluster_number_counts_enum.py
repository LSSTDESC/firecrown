"""Enumerations for cluster number counts likelihoods.
"""

from enum import Enum


class SupportedTracerNames(Enum):
    """Supported tracer names for cluster number counts likelihoods."""

    CLUSTER_COUNTS_TRUE_MASS = 1
    CLUSTER_COUNTS_RICHNESS_PROXY = 2
    CLUSTER_COUNTS_RICHNESS_PROXY_PLUSMEAN = 3
    CLUSTER_COUNTS_RICHNESS_MEANONLY_PROXY = 4

class SupportedDataTypes(Enum):
    """Supported data types for cluster number counts likelihoods."""

    CLUSTER_MASS_COUNT_WL = 1


class SupportedProxyTypes(Enum):
    """Supported proxy types for cluster number counts likelihoods.""" ""

    TRUE_MASS = 1
    RICHNESS_PROXY = 2

