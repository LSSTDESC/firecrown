"""Classes used to represent likelihoods and support functions.

Subpackages contain specific likelihood implementations,
e.g., Gaussian and Student-t.
The submodule :mod:`firecrown.likelihood._likelihood` contains
the abstract base class for likelihoods and likelihood loading utilities.

"""

# Core likelihood infrastructure
from firecrown.likelihood._likelihood import (
    Likelihood,
    NamedParameters,
    load_likelihood,
)

# Gaussian family likelihoods
from firecrown.likelihood._gaussian import ConstGaussian
from firecrown.likelihood._gaussfamily import GaussFamily, State
from firecrown.likelihood._student_t import StudentT

# Two-point statistics
from firecrown.likelihood._two_point import TwoPoint

# Cluster statistics
from firecrown.likelihood._binned_cluster import BinnedCluster
from firecrown.likelihood._binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown.likelihood._binned_cluster_number_counts_deltasigma import (
    BinnedClusterDeltaSigma,
)

# Supernova statistics
from firecrown.likelihood._supernova import Supernova

# Source infrastructure
from firecrown.likelihood._source import (
    Source,
    Tracer,
    SourceSystematic,
)

# Base statistic class
from firecrown.likelihood._statistic import Statistic

# Subpackages - make them accessible as module attributes
# This allows: import firecrown.likelihood.weak_lensing
# pylint: disable=unused-import
from . import number_counts
from . import factories
from . import weak_lensing
from . import supernova

# pylint: enable=unused-import

__all__ = [
    # Core likelihood infrastructure
    "Likelihood",
    "NamedParameters",
    "load_likelihood",
    # Gaussian family likelihoods
    "ConstGaussian",
    "GaussFamily",
    "State",
    "StudentT",
    # Two-point statistics
    "TwoPoint",
    # Cluster statistics
    "BinnedCluster",
    "BinnedClusterNumberCounts",
    "BinnedClusterDeltaSigma",
    # Supernova statistics
    "Supernova",
    # Source infrastructure
    "Source",
    "Tracer",
    "SourceSystematic",
    # Base statistic class
    "Statistic",
    # Subpackages
    "weak_lensing",
    "number_counts",
    "factories",
    "supernova",
]
