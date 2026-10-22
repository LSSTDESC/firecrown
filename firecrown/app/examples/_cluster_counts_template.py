"""Cluster number counts likelihood factory for Firecrown analysis.

This module defines the likelihood factory function for binned cluster number
counts and mean cluster mass, using richness as the mass proxy and
spectroscopic cluster redshifts.
"""

from pathlib import Path

import pyccl
from crow import ClusterAbundance, kernel, mass_proxy
from crow.properties import ClusterProperty
from crow.recipes.binned_exact import ExactBinnedClusterRecipe

from firecrown.likelihood import (
    BinnedClusterNumberCounts,
    ConstGaussian,
    NamedParameters,
)
from firecrown.likelihood.factories import load_sacc_data
from firecrown.modeling_tools import ModelingTools

# pylint: disable=duplicate-code


def get_cluster_recipe(
    pivot_mass: float = 14.625862906,
    pivot_redshift: float = 0.6,
    mass_interval: tuple[float, float] = (12.0, 17.0),
    true_z_interval: tuple[float, float] = (0.1, 2.0),
) -> ExactBinnedClusterRecipe:
    """Create the recipe that predicts binned cluster observables.

    The cosmology given to ClusterAbundance is only a placeholder; Firecrown
    replaces it with the sampled cosmology before every evaluation.

    :param pivot_mass: Pivot log10 mass of the Murata mass-richness relation
    :param pivot_redshift: Pivot redshift of the Murata mass-richness relation
    :param mass_interval: Range of true log10 mass to integrate over
    :param true_z_interval: Range of true redshift to integrate over
    :returns: The cluster recipe
    """
    cluster_theory = ClusterAbundance(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
    )
    return ExactBinnedClusterRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=mass_proxy.MurataBinned(pivot_mass, pivot_redshift),
        completeness=None,
        purity=None,
        mass_interval=mass_interval,
        true_z_interval=true_z_interval,
    )


def build_likelihood(params: NamedParameters) -> tuple[ConstGaussian, ModelingTools]:
    """Build a cluster number counts likelihood.

    Required parameters:
    - sacc_file: Path to SACC data file
    - survey_name: Name of the survey tracer in the SACC file

    Optional parameters:
    - use_cluster_counts: Include the cluster counts (default: True)
    - use_mean_log_mass: Include the mean log10 mass (default: False)

    :param params: Named parameters containing configuration
    :returns: Configured likelihood and modeling tools
    """
    sacc_file = Path(params.get_string("sacc_file"))
    survey_name = params.get_string("survey_name")

    # Select which cluster observables enter the data vector.
    average_on = ClusterProperty.NONE
    if params.get_bool("use_cluster_counts", True):
        average_on |= ClusterProperty.COUNTS
    if params.get_bool("use_mean_log_mass", False):
        average_on |= ClusterProperty.MASS

    likelihood = ConstGaussian(
        statistics=[
            BinnedClusterNumberCounts(average_on, survey_name, get_cluster_recipe())
        ]
    )

    assert sacc_file.exists(), f"SACC file not found: {sacc_file}"
    likelihood.read(load_sacc_data(sacc_file))

    # The halo mass function only needs the linear matter power spectrum.
    modeling_tools = ModelingTools()

    return likelihood, modeling_tools
