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

from firecrown.likelihood._binned_cluster import BinnedCluster
from firecrown.likelihood._binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown.likelihood._binned_cluster_number_counts_shear import (
    BinnedClusterShearProfile,
)
from firecrown.likelihood._cmb import CMBConvergence, CMBConvergenceArgs
from firecrown.likelihood._gaussfamily import GaussFamily, State
from firecrown.likelihood._gaussian import ConstGaussian
from firecrown.likelihood._gaussian_pointmass import ConstGaussianPM, PointMassData
from firecrown.likelihood._likelihood import (
    load_likelihood,
    load_likelihood_from_module_type,
)
from firecrown.likelihood._student_t import StudentT
from firecrown.likelihood._two_point import TwoPoint, TwoPointFactory
from firecrown.likelihood._updatable_wrapper import UpdatableClusterObjects
from firecrown.likelihood.supernova._supernova import Supernova
from firecrown.likelihood_base import (
    Likelihood,
    NamedParameters,
    Source,
    SourceGalaxy,
    SourceGalaxyArgs,
    SourceGalaxySystematic,
    SourceSystematic,
    Statistic,
    Tracer,
    TrivialStatistic,
)

from . import factories, number_counts, supernova, weak_lensing

__all__ = [
    "Likelihood",
    "NamedParameters",
    "load_likelihood",
    "load_likelihood_from_module_type",
    "ConstGaussian",
    "GaussFamily",
    "State",
    "StudentT",
    "UpdatableClusterObjects",
    "ConstGaussianPM",
    "PointMassData",
    "TwoPoint",
    "TwoPointFactory",
    "BinnedCluster",
    "BinnedClusterNumberCounts",
    "BinnedClusterShearProfile",
    "Supernova",
    "Source",
    "SourceGalaxy",
    "SourceGalaxyArgs",
    "CMBConvergence",
    "CMBConvergenceArgs",
    "SourceGalaxySystematic",
    "Tracer",
    "SourceSystematic",
    "Statistic",
    "TrivialStatistic",
    "weak_lensing",
    "number_counts",
    "factories",
    "supernova",
]
