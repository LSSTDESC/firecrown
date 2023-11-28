"""Tests for the binned cluster number counts module."""
from unittest.mock import Mock
import sacc
import pytest
import pyccl
from firecrown.models.cluster.integrator.integrator import Integrator
from firecrown.likelihood.gauss_family.statistic.source.source import SourceSystematic
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.parameters import ParamsMap
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.likelihood.gauss_family.statistic.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)


def test_create_binned_number_counts():
    integrator = Mock(spec=Integrator)
    integrator.integrate.return_value = 1.0
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "Test", integrator)
    assert bnc is not None
    assert bnc.properties == ClusterProperty.NONE
    assert bnc.survey_name == "Test"
    assert bnc.systematics == []
    assert bnc.theory_vector is None
    assert len(bnc.data_vector) == 0

    bnc = BinnedClusterNumberCounts(
        (ClusterProperty.COUNTS | ClusterProperty.MASS), "Test", integrator
    )
    assert ClusterProperty.COUNTS in bnc.properties
    assert ClusterProperty.MASS in bnc.properties

    systematics = [SourceSystematic("mock_systematic")]
    bnc = BinnedClusterNumberCounts(
        ClusterProperty.NONE, "Test", integrator, systematics
    )
    assert bnc.systematics == systematics


def test_get_data_vector():
    integrator = Mock(spec=Integrator)
    integrator.integrate.return_value = 1.0
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "Test", integrator)
    dv = bnc.get_data_vector()
    assert dv is not None
    assert len(dv) == 0


def test_read(cluster_sacc_data: sacc.Sacc):
    integrator = Mock(spec=Integrator)
    integrator.integrate.return_value = 1.0
    bnc = BinnedClusterNumberCounts(ClusterProperty.NONE, "my_survey", integrator)

    with pytest.raises(
        ValueError,
        match="You must specify at least one cluster property",
    ):
        bnc.read(cluster_sacc_data)

    bnc = BinnedClusterNumberCounts(ClusterProperty.COUNTS, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bin_limits) == 4
    assert len(bnc.data_vector) == 2
    assert len(bnc.sacc_indices) == 2

    bnc = BinnedClusterNumberCounts(ClusterProperty.MASS, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bin_limits) == 4
    assert len(bnc.data_vector) == 4
    assert len(bnc.sacc_indices) == 4

    bnc = BinnedClusterNumberCounts(
        (ClusterProperty.COUNTS | ClusterProperty.MASS), "my_survey", integrator
    )
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bin_limits) == 4
    assert len(bnc.data_vector) == 6
    assert len(bnc.sacc_indices) == 6


def test_compute_theory_vector(cluster_sacc_data: sacc.Sacc):
    integrator = Mock(spec=Integrator)
    integrator.integrate.return_value = 1.0
    tools = ModelingTools()

    hmf = pyccl.halos.MassFuncBocquet16()
    cosmo = pyccl.cosmology.CosmologyVanillaLCDM()
    params = ParamsMap()

    tools.cluster_abundance = ClusterAbundance(13, 17, 0, 2, hmf)
    tools.update(params)
    tools.prepare(cosmo)

    bnc = BinnedClusterNumberCounts(ClusterProperty.COUNTS, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 4

    bnc = BinnedClusterNumberCounts(ClusterProperty.MASS, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 4

    bnc = BinnedClusterNumberCounts(
        (ClusterProperty.COUNTS | ClusterProperty.MASS), "my_survey", integrator
    )
    bnc.read(cluster_sacc_data)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 8
