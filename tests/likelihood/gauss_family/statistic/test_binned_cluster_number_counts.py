"""Tests for the binned cluster number counts module."""

import sacc
import pytest
import pyccl
from crow import ClusterAbundance, ClusterShearProfile, kernel, mass_proxy
from crow.recipes.binned_exact import ExactBinnedClusterRecipe
from crow.recipes.binned_grid import GridBinnedClusterRecipe
from crow.properties import ClusterProperty
from firecrown.updatable import get_default_params_map
from firecrown.likelihood._source import SourceSystematic
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood._binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown.likelihood._binned_cluster_number_counts_shear import (
    BinnedClusterShearProfile,
)


def get_base_binned_exact(completeness, purity) -> ExactBinnedClusterRecipe:
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    cluster_recipe = ExactBinnedClusterRecipe(
        cluster_theory=ClusterAbundance(
            cosmo=pyccl.CosmologyVanillaLCDM(),
            halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        ),
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=mass_proxy.MurataBinned(pivot_mass, pivot_redshift),
        completeness=completeness,
        purity=purity,
        mass_interval=(13, 17),
        true_z_interval=(0, 2),
    )
    cluster_recipe.mass_distribution.parameters["mu0"] = 3.0
    cluster_recipe.mass_distribution.parameters["mu1"] = 0.86
    cluster_recipe.mass_distribution.parameters["mu2"] = 0.0
    cluster_recipe.mass_distribution.parameters["sigma0"] = 3.0
    cluster_recipe.mass_distribution.parameters["sigma1"] = 0.7
    cluster_recipe.mass_distribution.parameters["sigma2"] = 0.0
    return cluster_recipe


def get_base_binned_grid(
    completeness, purity, is_delta_sigma
) -> GridBinnedClusterRecipe:
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    cluster_recipe = GridBinnedClusterRecipe(
        cluster_theory=ClusterShearProfile(
            cosmo=pyccl.CosmologyVanillaLCDM(),
            halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
            is_delta_sigma=is_delta_sigma,
            use_beta_s_interp=True,
        ),
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=mass_proxy.MurataUnbinned(pivot_mass, pivot_redshift),
        completeness=completeness,
        purity=purity,
        mass_interval=(13, 17),
        true_z_interval=(0.1, 2),
        redshift_grid_size=20,
        mass_grid_size=50,
        proxy_grid_size=20,
    )
    cluster_recipe.mass_distribution.parameters["mu0"] = 3.0
    cluster_recipe.mass_distribution.parameters["mu1"] = 0.86
    cluster_recipe.mass_distribution.parameters["mu2"] = 0.0
    cluster_recipe.mass_distribution.parameters["sigma0"] = 3.0
    cluster_recipe.mass_distribution.parameters["sigma1"] = 0.7
    return cluster_recipe


@pytest.fixture(name="binned_exact")
def fixture_binned_exact() -> ExactBinnedClusterRecipe:
    return get_base_binned_exact(None, None)


@pytest.fixture(name="binned_grid_ds")
def fixture_binned_grid_ds() -> ExactBinnedClusterRecipe:
    return get_base_binned_grid(None, None, True)


@pytest.fixture(name="binned_grid_rs")
def fixture_binned_grid_rs() -> ExactBinnedClusterRecipe:
    recipe = get_base_binned_grid(None, None, False)
    recipe.cluster_theory.set_beta_parameters(10.0, 5.0)
    recipe.cluster_theory.set_beta_s_interp(0.1, 2.0)
    return recipe


def test_create_binned_number_counts(binned_exact: ExactBinnedClusterRecipe) -> None:
    recipe = binned_exact
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "Test", recipe)
    assert bnc is not None
    assert bnc.cluster_properties == ClusterProperty.NONE
    assert bnc.survey_name == "Test"
    assert bnc.systematics == []
    assert bnc.theory_vector is None
    assert len(bnc.data_vector) == 0

    bnc = BinnedClusterNumberCounts(
        (ClusterProperty.COUNTS | ClusterProperty.MASS), "Test", recipe
    )
    assert ClusterProperty.COUNTS in bnc.cluster_properties
    assert ClusterProperty.MASS in bnc.cluster_properties

    systematics = [SourceSystematic("mock_systematic")]
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "Test", recipe, systematics)
    assert bnc.systematics == systematics


def test_create_binned_shear_profile(binned_grid_ds: GridBinnedClusterRecipe) -> None:
    recipe = binned_grid_ds
    bnd = BinnedClusterShearProfile(ClusterProperty.NONE, "Test", recipe)
    assert bnd is not None
    assert bnd.cluster_properties == ClusterProperty.NONE
    assert bnd.survey_name == "Test"
    assert bnd.systematics == []
    assert bnd.theory_vector is None
    assert len(bnd.data_vector) == 0

    bnd = BinnedClusterShearProfile((ClusterProperty.DELTASIGMA), "Test", recipe)
    assert ClusterProperty.DELTASIGMA in bnd.cluster_properties

    bnd = BinnedClusterShearProfile((ClusterProperty.SHEAR), "Test", recipe)
    assert ClusterProperty.SHEAR in bnd.cluster_properties

    systematics = [SourceSystematic("mock_systematic")]
    bnd = BinnedClusterShearProfile(ClusterProperty.NONE, "Test", recipe, systematics)
    assert bnd.systematics == systematics


def test_get_data_vector(
    binned_exact: ExactBinnedClusterRecipe, binned_grid_ds: GridBinnedClusterRecipe
) -> None:
    recipe = binned_exact
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "Test", recipe)
    bnd = BinnedClusterShearProfile(ClusterProperty.NONE, "Test", recipe)
    dv = bnc.get_data_vector()
    dvd = bnd.get_data_vector()
    assert dv is not None
    assert len(dv) == 0
    assert dvd is not None
    assert len(dvd) == 0
    recipe = binned_grid_ds
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "Test", recipe)
    bnd = BinnedClusterShearProfile(ClusterProperty.NONE, "Test", recipe)
    dv = bnc.get_data_vector()
    dvd = bnd.get_data_vector()
    assert dv is not None
    assert len(dv) == 0
    assert dvd is not None
    assert len(dvd) == 0


def test_read_throws_if_no_property(
    cluster_sacc_data: sacc.Sacc, binned_exact: ExactBinnedClusterRecipe
) -> None:
    recipe = binned_exact
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "my_survey", recipe)

    with pytest.raises(
        ValueError,
        match="You must specify at least one cluster property",
    ):
        bnc.read(cluster_sacc_data)
    bnd = BinnedClusterShearProfile(ClusterProperty.NONE, "my_survey", recipe)

    with pytest.raises(
        ValueError,
        match="You must specify at least one cluster property",
    ):
        bnd.read(cluster_sacc_data)


def test_read_single_property(
    cluster_sacc_data: sacc.Sacc, binned_exact: ExactBinnedClusterRecipe
) -> None:
    recipe = binned_exact

    bnc = BinnedClusterNumberCounts(ClusterProperty.COUNTS, "my_survey", recipe)
    bnd = BinnedClusterShearProfile(ClusterProperty.DELTASIGMA, "my_survey", recipe)
    bns = BinnedClusterShearProfile(ClusterProperty.DELTASIGMA, "my_survey", recipe)
    bnc.read(cluster_sacc_data)
    bnd.read(cluster_sacc_data)
    bns.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert bnd.sky_area == 4000
    assert bns.sky_area == 4000
    assert len(bnc.bins) == 2
    assert len(bnc.bins) == 2
    assert len(bns.bins) == 2
    assert len(bnc.data_vector) == 2
    assert len(bnd.data_vector) == 2
    assert len(bns.data_vector) == 2
    assert bnc.sacc_indices is not None
    assert bnd.sacc_indices is not None
    assert bns.sacc_indices is not None
    assert len(bnc.sacc_indices) == 2
    assert len(bnd.sacc_indices) == 2
    assert len(bns.sacc_indices) == 2
    bnc = BinnedClusterNumberCounts(ClusterProperty.MASS, "my_survey", recipe)
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bins) == 2
    assert len(bnc.data_vector) == 2
    assert bnc.sacc_indices is not None
    assert len(bnc.sacc_indices) == 2


def test_read_multiple_properties(
    cluster_sacc_data: sacc.Sacc, binned_exact: ExactBinnedClusterRecipe
) -> None:
    recipe = binned_exact
    bnc = BinnedClusterNumberCounts(
        (ClusterProperty.COUNTS | ClusterProperty.MASS), "my_survey", recipe
    )
    bns = BinnedClusterShearProfile(
        ClusterProperty.SHEAR | ClusterProperty.DELTASIGMA, "my_survey", recipe
    )
    bnc.read(cluster_sacc_data)
    bns.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert bns.sky_area == 4000
    assert len(bnc.bins) == 2
    assert len(bns.bins) == 2
    assert len(bnc.data_vector) == 4
    assert len(bns.data_vector) == 4
    assert bnc.sacc_indices is not None
    assert bns.sacc_indices is not None
    assert len(bnc.sacc_indices) == 4
    assert len(bns.sacc_indices) == 4


def test_compute_theory_vector(
    cluster_sacc_data: sacc.Sacc,
    binned_exact: ExactBinnedClusterRecipe,
    binned_grid_ds: GridBinnedClusterRecipe,
    binned_grid_rs: GridBinnedClusterRecipe,
) -> None:
    recipe = binned_exact
    recipe_ds = binned_grid_ds
    recipe_rs = binned_grid_rs
    tools = ModelingTools()

    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    bnc = BinnedClusterNumberCounts(ClusterProperty.COUNTS, "my_survey", recipe)
    bnc.read(cluster_sacc_data)
    params = get_default_params_map(bnc.updatable_parameters)
    bnc.update(params)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 2

    bnc = BinnedClusterNumberCounts(ClusterProperty.MASS, "my_survey", recipe)
    bnc.read(cluster_sacc_data)
    bnc.update(params)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 2

    bnc = BinnedClusterNumberCounts(
        (ClusterProperty.COUNTS | ClusterProperty.MASS), "my_survey", recipe
    )
    bnc.read(cluster_sacc_data)
    bnc.update(params)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 4

    bnc = BinnedClusterNumberCounts(
        (ClusterProperty.COUNTS | ClusterProperty.MASS | ClusterProperty.DELTASIGMA),
        "my_survey",
        recipe,
    )
    bnc.read(cluster_sacc_data)
    bnc.update(params)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 4

    bnd = BinnedClusterShearProfile(ClusterProperty.DELTASIGMA, "my_survey", recipe_ds)
    bnd.read(cluster_sacc_data)
    params.params["cluster_theory.cluster_concentration"] = 4.0
    bnd.update(params)
    tv = bnd.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 2

    bnd = BinnedClusterShearProfile(
        (ClusterProperty.COUNTS | ClusterProperty.DELTASIGMA), "my_survey", recipe_ds
    )
    bnd.read(cluster_sacc_data)
    bnd.update(params)
    tv = bnd.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 2
    bns = BinnedClusterShearProfile(ClusterProperty.DELTASIGMA, "my_survey", recipe_rs)
    bns.read(cluster_sacc_data)
    bns.update(params)
    tv = bns.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 2

    bns = BinnedClusterShearProfile(
        (ClusterProperty.COUNTS | ClusterProperty.DELTASIGMA), "my_survey", recipe_rs
    )
    bns.read(cluster_sacc_data)
    bns.update(params)
    tv = bns.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 2
