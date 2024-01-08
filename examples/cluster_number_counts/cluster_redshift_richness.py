"""Likelihood factory function for cluster number counts."""

import os
from typing import Tuple

import pyccl as ccl
import sacc

from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.statistic.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.murata_binned_spec_z import (
    MurataBinnedSpecZRecipe,
)
from firecrown.models.cluster.recipes.true_mass_spec_z import TrueMassSpecZRecipe


def get_cluster_abundance() -> ClusterAbundance:
    hmf = ccl.halos.MassFuncBocquet16()
    min_mass, max_mass = 13.0, 16.0
    min_z, max_z = 0.2, 0.8
    cluster_abundance = ClusterAbundance(min_mass, max_mass, min_z, max_z, hmf)

    return cluster_abundance


def build_likelihood(
    build_parameters: NamedParameters,
) -> Tuple[Likelihood, ModelingTools]:
    """
    Here we instantiate the number density (or mass function) object.
    """

    # Pull params for the likelihood from build_parameters
    average_on = ClusterProperty.NONE
    if build_parameters.get_bool("use_cluster_counts", True):
        average_on |= ClusterProperty.COUNTS
    if build_parameters.get_bool("use_mean_log_mass", False):
        average_on |= ClusterProperty.MASS

    survey_name = "numcosmo_simulated_redshift_richness"
    likelihood = ConstGaussian(
        [BinnedClusterNumberCounts(average_on, survey_name, MurataBinnedSpecZRecipe())]
        # [BinnedClusterNumberCounts(average_on, survey_name, TrueMassSpecZRecipe())]
    )

    # Read in sacc data
    sacc_file_nm = "cluster_redshift_richness_sacc_data.fits"
    sacc_path = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/cluster_number_counts/")
    )
    sacc_data = sacc.Sacc.load_fits(os.path.join(sacc_path, sacc_file_nm))
    likelihood.read(sacc_data)

    cluster_abundance = get_cluster_abundance()
    modeling_tools = ModelingTools(cluster_abundance=cluster_abundance)

    return likelihood, modeling_tools
