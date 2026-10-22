"""Cluster number counts and weak lensing likelihood factory for Firecrown.

This module defines the likelihood factory function for binned cluster number
counts combined with stacked weak lensing profiles, either the excess surface
density DeltaSigma or the reduced tangential shear.
"""

from pathlib import Path

import pyccl
from crow import ClusterShearProfile, kernel, mass_proxy
from crow.integrator.numcosmo_integrator import (
    NumCosmoIntegralMethod,
    NumCosmoIntegrator,
)
from crow.properties import ClusterProperty
from crow.recipes.binned_exact import ExactBinnedClusterRecipe
from crow.recipes.binned_grid import GridBinnedClusterRecipe

from firecrown.likelihood import (
    BinnedClusterNumberCounts,
    BinnedClusterShearProfile,
    ConstGaussian,
    NamedParameters,
)
from firecrown.likelihood.factories import load_sacc_data
from firecrown.modeling_tools import ModelingTools

# pylint: disable=duplicate-code

PIVOT_MASS = 14.625862906
PIVOT_REDSHIFT = 0.6
MASS_INTERVAL = (12.0, 17.0)
TRUE_Z_INTERVAL = (0.1, 2.0)


def get_delta_sigma_recipe() -> ExactBinnedClusterRecipe:
    """Create the recipe that predicts binned counts and DeltaSigma profiles.

    The cosmology given to ClusterShearProfile is only a placeholder; Firecrown
    replaces it with the sampled cosmology before every evaluation.

    :returns: The cluster recipe
    """
    cluster_theory = ClusterShearProfile(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        cluster_concentration=None,
        is_delta_sigma=True,
        use_beta_s_interp=True,
    )
    recipe = ExactBinnedClusterRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=mass_proxy.MurataBinned(PIVOT_MASS, PIVOT_REDSHIFT),
        completeness=None,
        purity=None,
        mass_interval=MASS_INTERVAL,
        true_z_interval=TRUE_Z_INTERVAL,
    )
    # The crow default, p-adaptive cubature, converges to a wrong value for some
    # DeltaSigma bins; h-adaptive cubature does not.
    recipe.integrator = NumCosmoIntegrator(method=NumCosmoIntegralMethod.H_V)
    return recipe


def get_reduced_shear_recipe() -> GridBinnedClusterRecipe:
    """Create the recipe that predicts binned counts and reduced shear profiles.

    The mean lensing efficiency of the sources is tabulated once over the true
    redshift interval, for sources up to z = 5 and z_inf = 10, using the
    placeholder cosmology.

    :returns: The cluster recipe
    """
    cluster_theory = ClusterShearProfile(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        cluster_concentration=None,
        is_delta_sigma=False,
        use_beta_s_interp=True,
    )
    cluster_theory.set_beta_parameters(10.0, 5.0)
    cluster_theory.set_beta_s_interp(*TRUE_Z_INTERVAL)
    return GridBinnedClusterRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=mass_proxy.MurataUnbinned(PIVOT_MASS, PIVOT_REDSHIFT),
        completeness=None,
        purity=None,
        mass_interval=MASS_INTERVAL,
        true_z_interval=TRUE_Z_INTERVAL,
    )


def build_likelihood(params: NamedParameters) -> tuple[ConstGaussian, ModelingTools]:
    """Build a cluster counts and weak lensing likelihood.

    Required parameters:
    - sacc_file: Path to SACC data file
    - survey_name: Name of the survey tracer in the SACC file
    - profile: Lensing profile, either "delta_sigma" or "reduced_shear"

    Optional parameters:
    - use_cluster_counts: Include the cluster counts (default: True)
    - use_mean_log_mass: Include the mean log10 mass (default: False)

    :param params: Named parameters containing configuration
    :returns: Configured likelihood and modeling tools
    """
    sacc_file = Path(params.get_string("sacc_file"))
    survey_name = params.get_string("survey_name")
    profile = params.get_string("profile")

    # Select which cluster observables enter the data vector.
    average_on = ClusterProperty.NONE
    if params.get_bool("use_cluster_counts", True):
        average_on |= ClusterProperty.COUNTS
    if params.get_bool("use_mean_log_mass", False):
        average_on |= ClusterProperty.MASS

    recipe: ExactBinnedClusterRecipe | GridBinnedClusterRecipe
    match profile:
        case "delta_sigma":
            average_on |= ClusterProperty.DELTASIGMA
            recipe = get_delta_sigma_recipe()
        case "reduced_shear":
            average_on |= ClusterProperty.SHEAR
            recipe = get_reduced_shear_recipe()
        case _:
            raise ValueError(f"Unknown lensing profile: {profile}")

    likelihood = ConstGaussian(
        statistics=[
            BinnedClusterNumberCounts(average_on, survey_name, recipe),
            BinnedClusterShearProfile(average_on, survey_name, recipe),
        ]
    )

    assert sacc_file.exists(), f"SACC file not found: {sacc_file}"
    likelihood.read(load_sacc_data(sacc_file))

    # The halo mass function only needs the linear matter power spectrum.
    modeling_tools = ModelingTools()

    return likelihood, modeling_tools
