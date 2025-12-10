"""Likelihood factory function for cluster number counts."""

import os
import pyccl

from crow import ClusterShearProfile, kernel, mass_proxy
from crow.properties import ClusterProperty
from crow.recipes.binned_exact import ExactBinnedClusterRecipe

from firecrown.likelihood.factories import load_sacc_data
from firecrown.likelihood import (
    BinnedClusterNumberCounts,
    ConstGaussian,
    Likelihood,
    NamedParameters,
)
from firecrown.modeling_tools import ModelingTools


def get_cluster_shear_profile() -> ClusterShearProfile:
    """
    Creates and returns a ClusterShearProfile object with the same
    configuration as in your previous code snippet.
    """

    cluster_theory = ClusterShearProfile(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        cluster_concentration=None,
        is_delta_sigma=True,
        use_beta_s_interp=True,
    )

    return cluster_theory


def get_cluster_recipe(
    cluster_theory=None,
    pivot_mass: float = 14.625862906,
    pivot_redshift: float = 0.6,
    mass_interval=(12, 17),
    true_z_interval=(0.1, 2.0),
):
    """
    Creates and returns an ExactBinnedClusterRecipe.
    Parameters
    ----------
    cluster_theory : ClusterShearProfile or None
        If None, uses get_cluster_shear_profile()

    Returns
    -------
    ExactBinnedClusterRecipe
    """

    if cluster_theory is None:
        cluster_theory = get_cluster_shear_profile()

    redshift_distribution = kernel.SpectroscopicRedshift()
    mass_distribution = mass_proxy.MurataBinned(pivot_mass, pivot_redshift)

    recipe = ExactBinnedClusterRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=redshift_distribution,
        mass_distribution=mass_distribution,
        completeness=None,
        purity=None,
        mass_interval=mass_interval,
        true_z_interval=true_z_interval,
    )

    return recipe


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

    survey_name = "numcosmo_sim_red_richness_shear_dsigma"

    recipe = get_cluster_recipe()

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
    sacc_file_nm = "cluster_redshift_richness_shear_dsigma_sacc.fits"
    sacc_path = os.path.expanduser(
        os.path.expandvars("${FIRECROWN_DIR}/examples/cluster_number_counts/")
    )
    sacc_data = load_sacc_data(os.path.join(sacc_path, sacc_file_nm))
    likelihood.read(sacc_data)

    modeling_tools = ModelingTools()

    return likelihood, modeling_tools
