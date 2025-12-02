"""Likelihood implementations and infrastructure.

This module provides the core likelihood framework and specific likelihood
implementations for various cosmological observables.

Core Components:
    - :class:`Likelihood`: Abstract base class for all likelihoods
    - :class:`Statistic`: Base class for observable statistics
    - :class:`Source`: Infrastructure for data sources and systematics
    - Likelihood loading utilities

Likelihood Types:
    - :class:`ConstGaussian`: Constant covariance Gaussian likelihood
    - :class:`GaussFamily`: Gaussian family likelihoods
    - :class:`StudentT`: Student-t distributed likelihood
    - :class:`TwoPoint`: Two-point correlation statistics
    - :class:`BinnedCluster`: Binned cluster statistics base class
    - :class:`BinnedClusterNumberCounts`: Cluster abundance likelihoods
    - :class:`BinnedClusterShearProfile`: Cluster weak lensing likelihoods
    - :class:`Supernova`: Supernova distance modulus likelihoods
    - :class:`UpdatableClusterObjects`: Connector for creating Firecrown
    - cluster updatables from external code

Subpackages:
    - :mod:`weak_lensing`: Weak lensing systematics and sources
    - :mod:`number_counts`: Galaxy number counts systematics and sources
    - :mod:`supernova`: Supernova-specific implementations
    - :mod:`factories`: Factory functions for creating likelihood components
"""

# Third-party imports
from numcosmo_py import Ncm

# Core likelihood infrastructure - base classes from _base.py
from firecrown.likelihood._base import (
    Likelihood,
    NamedParameters,
    Source,
    SourceGalaxyArgs,
    SourceGalaxySystematic,
    SourceSystematic,
    Statistic,
    Tracer,
    TrivialStatistic,
)
from firecrown.likelihood._likelihood import (
    load_likelihood,
    load_likelihood_from_module_type,
)

# Gaussian family likelihoods
from firecrown.likelihood._gaussian import ConstGaussian
from firecrown.likelihood._gaussfamily import GaussFamily, State
from firecrown.likelihood._student_t import StudentT
from firecrown.likelihood._gaussian_pointmass import ConstGaussianPM
from firecrown.likelihood._updatable_wrapper import UpdatableClusterObjects

# Two-point statistics
from firecrown.likelihood._two_point import TwoPoint, TwoPointFactory

# Cluster statistics
from firecrown.likelihood._binned_cluster import BinnedCluster
from firecrown.likelihood._binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown.likelihood._binned_cluster_number_counts_deltasigma import (
    BinnedClusterShearProfile,
)

# Supernova statistics
from firecrown.likelihood.supernova._supernova import Supernova

# Source and Statistic classes now imported from _base.py above

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
    "load_likelihood_from_module_type",
    # Gaussian family likelihoods
    "ConstGaussian",
    "GaussFamily",
    "State",
    "StudentT",
    "ConstGaussianPM",
    "UpdatableClusterObjects",
    # Two-point statistics
    "TwoPoint",
    "TwoPointFactory",
    # Cluster statistics
    "BinnedCluster",
    "BinnedClusterNumberCounts",
    "BinnedClusterShearProfile",
    # Supernova statistics
    "Supernova",
    # Source infrastructure
    "Source",
    "SourceGalaxyArgs",
    "SourceGalaxySystematic",
    "Tracer",
    "SourceSystematic",
    # Base statistic class
    "Statistic",
    "TrivialStatistic",
    # Subpackages
    "weak_lensing",
    "number_counts",
    "factories",
    "supernova",
]
