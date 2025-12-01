"""Likelihood factory function for cluster number counts."""

import os
import sys

import pyccl
import sacc
# remove this line after crow becomes installable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crow import ClusterShearProfile, kernel, mass_proxy
from crow.recipes.murata_binned_spec_z import MurataBinnedSpecZRecipe

from firecrown.likelihood import (
    ConstGaussian,
    BinnedClusterDeltaSigma,
    BinnedClusterNumberCounts,
    Likelihood,
    NamedParameters,
)
from firecrown.likelihood.binned_cluster_number_counts_deltasigma import (
    BinnedClusterShearProfile,
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster import (
    ClusterAbundance,
    ClusterDeltaSigma,
    ClusterProperty,
)


def build_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """Builds the likelihood for Firecrown."""
    # Pull params for the likelihood from build_parameters
    average_on = ClusterProperty.NONE
    if build_parameters.get_bool("use_cluster_counts", True):
        average_on |= ClusterProperty.COUNTS
    if build_parameters.get_bool("use_mean_log_mass", True):
        average_on |= ClusterProperty.MASS
    if build_parameters.get_bool("use_mean_deltasigma", True):
        average_on |= ClusterProperty.DELTASIGMA

    redshift_distribution = kernel.SpectroscopicRedshift()
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    mass_distribution = mass_proxy.MurataBinned(pivot_mass, pivot_redshift)
    survey_name = "numcosmo_simulated_redshift_richness_deltasigma"

    cluster_theory = ClusterShearProfile(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        cluster_concentration=4.0,
        is_delta_sigma=True,
        use_beta_s_interp=True,
    )
    recipe = MurataBinnedSpecZRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=redshift_distribution,
        mass_distribution=mass_distribution,
        completeness=None,
        mass_interval=(12, 17),
        true_z_interval=(0.1, 2.0),
    )
    likelihood = ConstGaussian(
        [
            BinnedClusterNumberCounts(
                average_on,
                survey_name,
                recipe,
            ),
            BinnedClusterShearProfile(
                average_on,
                survey_name,
                recipe,
            ),
        ]
    )

    # Read in sacc data
    sacc_file_nm = "cluster_redshift_richness_deltasigma_sacc_data.fits"
    sacc_data = sacc.Sacc.load_fits(sacc_file_nm)
    likelihood.read(sacc_data)

    modeling_tools = ModelingTools()

    return likelihood, modeling_tools
