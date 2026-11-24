"""Binned galaxy cluster statistics and their factories.

Subpackages:
    - :mod:`abundance`: number counts and mean mass in redshift-richness bins
    - :mod:`lensing`: stacked excess surface mass density or reduced shear

A :class:`ClusterModelFactory` describes the crow model of one cluster sample;
:class:`BinnedClusterFactory` combines the abundance and lensing factories of
all samples, as :class:`~firecrown.likelihood.TwoPointFactory` combines the
factories of the two-point tracers.
"""

from firecrown.likelihood.clusters._binned_cluster import BinnedCluster
from firecrown.likelihood.clusters._factories import (
    ClusterModelFactory,
    CompletenessAguena16Factory,
    ExactBinnedClusterRecipeFactory,
    GridBinnedClusterRecipeFactory,
    HaloMassFunctionFactory,
    MurataBinnedFactory,
    MurataUnbinnedFactory,
    PurityAguena16Factory,
    PurityAguena16LnProxyFactory,
    SpectroscopicRedshiftFactory,
)
from firecrown.likelihood.clusters._updatable_wrapper import UpdatableClusterObjects
from firecrown.likelihood.clusters.abundance import (
    BinnedClusterNumberCounts,
    BinnedClusterNumberCountsFactory,
)
from firecrown.likelihood.clusters.lensing import (
    BinnedClusterShearProfile,
    BinnedClusterShearProfileFactory,
    LensingEfficiencyOptions,
)
from firecrown.likelihood.clusters._binned_cluster_factory import (
    BinnedClusterFactory,
)

from . import abundance, lensing

__all__ = [
    # Statistics
    "BinnedCluster",
    "BinnedClusterNumberCounts",
    "BinnedClusterShearProfile",
    "UpdatableClusterObjects",
    # Factories
    "BinnedClusterFactory",
    "BinnedClusterNumberCountsFactory",
    "BinnedClusterShearProfileFactory",
    "ClusterModelFactory",
    "CompletenessAguena16Factory",
    "ExactBinnedClusterRecipeFactory",
    "GridBinnedClusterRecipeFactory",
    "HaloMassFunctionFactory",
    "LensingEfficiencyOptions",
    "MurataBinnedFactory",
    "MurataUnbinnedFactory",
    "PurityAguena16Factory",
    "PurityAguena16LnProxyFactory",
    "SpectroscopicRedshiftFactory",
    # Subpackages
    "abundance",
    "lensing",
]
