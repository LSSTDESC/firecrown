"""Likelihood factory function for cluster number counts."""

import os
import sys

import pyccl
import sacc
# remove this line after crow becomes installable
sys.path.append("/sps/lsst/users/ebarroso/crow/crow/")
from crow import ClusterShearProfile, kernel, mass_proxy
from crow.properties import ClusterProperty
from crow.recipes.binned_grid import GridBinnedClusterRecipe
from crow.recipes.binned_exact import ExactBinnedClusterRecipe

from firecrown.likelihood import (
    ConstGaussian,
    BinnedClusterShearProfile,
    BinnedClusterNumberCounts,
    Likelihood,
    NamedParameters,
)   

from firecrown.modeling_tools import ModelingTools


def get_cluster_abundance() -> ClusterAbundance:
    """Creates and returns a ClusterAbundance object."""
    hmf = ccl.halos.MassFuncBocquet16()
    min_mass, max_mass = 13.0, 17.0
    min_z, max_z = 0.2, 0.8
    cluster_abundance = ClusterAbundance((min_mass, max_mass), (min_z, max_z), hmf)

    return cluster_abundance


def build_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """Builds the likelihood for Firecrown."""
    # Pull params for the likelihood from build_parameters
    average_on = ClusterProperty.NONE
    if build_parameters.get_bool("use_cluster_counts", True):
        average_on |= ClusterProperty.COUNTS
    if build_parameters.get_bool("use_mean_log_mass", False):
        average_on |= ClusterProperty.MASS

    survey_name = "SDSSCluster_redshift_richness"
    likelihood = ConstGaussian(
        [BinnedClusterNumberCounts(average_on, survey_name, MurataBinnedSpecZRecipe())]
    )

    # Read in sacc data
    sacc_file_nm = "SDSSTEST_redshift_richness_sacc_data.fits"
    sacc_path = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/cluster_number_counts/")
    )
    sacc_data = sacc.Sacc.load_fits(os.path.join(sacc_path, sacc_file_nm))
    likelihood.read(sacc_data)

    cluster_abundance = get_cluster_abundance()
    modeling_tools = ModelingTools(cluster_abundance=cluster_abundance)

    return likelihood, modeling_tools
