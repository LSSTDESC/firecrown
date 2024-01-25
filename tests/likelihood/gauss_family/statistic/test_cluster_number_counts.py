"""Tests for the binned cluster number counts module."""
from unittest.mock import Mock
import sacc
import pytest
import pyccl
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe
from firecrown.likelihood.gauss_family.statistic.source.source import SourceSystematic
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.parameters import ParamsMap
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.likelihood.gauss_family.statistic.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)


def test_create_binned_number_counts():
    recipe = Mock(spec=ClusterRecipe)
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


def test_get_data_vector():
    recipe = Mock(spec=ClusterRecipe)
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "Test", recipe)
    dv = bnc.get_data_vector()
    assert dv is not None
    assert len(dv) == 0


def test_read_throws_if_no_property(cluster_sacc_data: sacc.Sacc):
    recipe = Mock(spec=ClusterRecipe)
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "my_survey", recipe)

    with pytest.raises(
        ValueError,
        match="You must specify at least one cluster property",
    ):
        bnc.read(cluster_sacc_data)


def test_read_single_property(cluster_sacc_data: sacc.Sacc):
    recipe = Mock(spec=ClusterRecipe)

    bnc = BinnedClusterNumberCounts(ClusterProperty.COUNTS, "my_survey", recipe)
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bins) == 2
    assert len(bnc.data_vector) == 2
    assert bnc.sacc_indices is not None
    assert len(bnc.sacc_indices) == 2

    bnc = BinnedClusterNumberCounts(ClusterProperty.MASS, "my_survey", recipe)
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bins) == 2
    assert len(bnc.data_vector) == 2
    assert bnc.sacc_indices is not None
    assert len(bnc.sacc_indices) == 2


def test_read_multiple_properties(cluster_sacc_data: sacc.Sacc):
    recipe = Mock(spec=ClusterRecipe)
    bnc = BinnedClusterNumberCounts(
        (ClusterProperty.COUNTS | ClusterProperty.MASS), "my_survey", recipe
    )
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bins) == 2
    assert len(bnc.data_vector) == 4
    assert bnc.sacc_indices is not None
    assert len(bnc.sacc_indices) == 4


def test_compute_theory_vector(cluster_sacc_data: sacc.Sacc):
    recipe = Mock(spec=ClusterRecipe)
    recipe.evaluate_theory_prediction.return_value = 1.0
    tools = ModelingTools()

    hmf = pyccl.halos.MassFuncBocquet16()
    cosmo = pyccl.cosmology.CosmologyVanillaLCDM()
    params = ParamsMap()

    tools.cluster_abundance = ClusterAbundance(13, 17, 0, 2, hmf)
    tools.update(params)
    tools.prepare(cosmo)

    bnc = BinnedClusterNumberCounts(ClusterProperty.COUNTS, "my_survey", recipe)
    bnc.read(cluster_sacc_data)
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
