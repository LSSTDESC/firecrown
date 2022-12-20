from enum import Enum


class SupportedTracerNames(Enum):
    cluster_counts_true_mass = 1
    cluster_counts_richness_proxy = 2


class SupportedDataTypes(Enum):
    cluster_mass_count_wl = 1


class SupportedProxyTypes(Enum):
    true_mass = 1
    richness_proxy = 2
