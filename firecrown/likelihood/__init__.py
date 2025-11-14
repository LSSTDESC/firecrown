"""Classes used to represent likelihoods and support functions.

Subpackages contain specific likelihood implementations,
e.g., Gaussian and Student-t.
The submodule :mod:`firecrown.likelihood._likelihood` contains
the abstract base class for likelihoods and likelihood loading utilities.

"""

# Third-party imports
from numcosmo_py import Ncm

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

# Compatibility layer for NumCosmo < 0.27
# NumCosmo knowns aboud the old internal organization of Firecrown.
if not Ncm.cfg_version_check(0, 27, 0):  # pragma: no branch
    import types
    import sys

    # Create a new module 'likelihood.likelihood' for backwards compatibility
    # with older NumCosmo versions
    likelihood = types.ModuleType("likelihood.likelihood")
    # Add NamedParameters class to the module, ignoring type checking since it
    # is added dynamically
    likelihood.NamedParameters = NamedParameters  # type: ignore[attr-defined]
    # Register the module in sys.modules to make it importable
    sys.modules["firecrown.likelihood.likelihood"] = likelihood

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
