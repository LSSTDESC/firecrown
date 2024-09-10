"""Tests for the cluster data module."""

import pytest
import sacc
from firecrown.models.cluster.abundance_data import AbundanceData
from firecrown.models.cluster.deltasigma_data import DeltaSigmaData
from firecrown.models.cluster.properties import ClusterProperty


def test_create_abundance_data():
    s = sacc.Sacc()
    ad = AbundanceData(s)
    dsd = DeltaSigmaData(s)

    # pylint: disable=protected-access
    assert ad._mass_index == 2
    assert dsd._mass_index == 2
    # pylint: disable=protected-access
    assert ad._redshift_index == 1
    assert dsd._redshift_index == 1
    # pylint: disable=protected-access
    assert ad._survey_index == 0
    assert dsd._survey_index == 0
    # pylint: disable=protected-access
    assert dsd._radius_index == 3

def test_get_survey_tracer_missing_survey_name(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data)
    dsd = DeltaSigmaData(cluster_sacc_data)
    data_list = [ad, dsd]
    for data_obj in data_list:
        with pytest.raises(
            ValueError,
            match="The SACC file does not contain the SurveyTracer the_black_lodge.",
        ):
            data_obj.get_survey_tracer("the_black_lodge")


def test_get_survey_tracer_wrong_tracer_type(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data)
    dsd = DeltaSigmaData(cluster_sacc_data)
    data_list = [ad, dsd]
    for data_obj in data_list:
        with pytest.raises(
            ValueError,
            match="The SACC tracer z_bin_tracer_1 is not a SurveyTracer.",
        ):
            data_obj.get_survey_tracer("z_bin_tracer_1")


def test_get_survey_tracer_returns_survey_tracer(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data)
    dsd = DeltaSigmaData(cluster_sacc_data)
    data_list = [ad, dsd]
    for data_obj in data_list:
        survey = data_obj.get_survey_tracer("my_survey")
        assert survey is not None
        assert isinstance(survey, sacc.tracers.SurveyTracer)


def test_get_bin_edges_cluster_counts(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data)
    bins = ad.get_bin_edges("my_survey", ClusterProperty.COUNTS)
    assert len(bins) == 2


def test_get_bin_edges_cluster_mass(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data)
    bins = ad.get_bin_edges("my_survey", ClusterProperty.MASS)
    assert len(bins) == 2

def test_get_bin_edges_cluster_radius(cluster_sacc_data: sacc.Sacc):
    dsd = DeltaSigmaData(cluster_sacc_data)
    bins = dsd.get_bin_edges("my_survey", ClusterProperty.DELTASIGMA)
    assert len(bins) == 2

def test_get_bin_edges_counts_and_mass(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data)
    bins = ad.get_bin_edges(
        "my_survey", (ClusterProperty.MASS | ClusterProperty.COUNTS)
    )
    assert len(bins) == 2


def test_get_bin_edges_not_implemented_cluster_property_throws(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data)
    with pytest.raises(NotImplementedError):
        ad.get_bin_edges("my_survey", ClusterProperty.SHEAR)


def test_observed_data_and_indices_by_survey_cluster_counts(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data)
    data, indices = ad.get_observed_data_and_indices_by_survey(
        "my_survey", ClusterProperty.COUNTS
    )
    assert len(data) == 2
    assert len(indices) == 2


def test_observed_data_and_indices_by_survey_cluster_mass(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data)
    data, indices = ad.get_observed_data_and_indices_by_survey(
        "my_survey", ClusterProperty.MASS
    )
    assert len(data) == 2
    assert len(indices) == 2

def test_observed_data_and_indices_by_survey_cluster_deltasigma(
    cluster_sacc_data: sacc.Sacc,
):
    dsd = DeltaSigmaData(cluster_sacc_data)
    data, indices = dsd.get_observed_data_and_indices_by_survey(
        "my_survey", ClusterProperty.DELTASIGMA
    )
    assert len(data) == 2
    assert len(indices) == 2


def test_observed_data_and_indices_by_survey_cluster_counts_and_mass(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data)
    data, indices = ad.get_observed_data_and_indices_by_survey(
        "my_survey", ClusterProperty.MASS | ClusterProperty.COUNTS
    )
    assert len(data) == 4
    assert len(indices) == 4

def test_observed_data_and_indices_by_survey_not_implemented_throws(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data)
    with pytest.raises(NotImplementedError):
        ad.get_observed_data_and_indices_by_survey("my_survey", ClusterProperty.SHEAR)


def test_observed_data_and_indices_no_data_throws():
    # pylint: disable=no-member
    cc = sacc.standard_types.cluster_counts

    s = sacc.Sacc()
    s.add_tracer("survey", "my_survey", 4000)
    s.add_tracer("bin_z", "z_bin_tracer_1", 0, 2)
    s.add_tracer("bin_z", "z_bin_tracer_2", 2, 4)
    s.add_tracer("bin_richness", "mass_bin_tracer_1", 0, 2)
    s.add_tracer("bin_richness", "mass_bin_tracer_2", 2, 4)

    s.add_data_point(
        cc,
        ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_1"),
        1,
    )
    s.add_data_point(
        cc,
        ("my_survey", "z_bin_tracer_1", "mass_bin_tracer_2"),
        1,
    )

    ad = AbundanceData(s)

    with pytest.raises(
        ValueError, match="The SACC file does not contain any tracers for the"
    ):
        ad.get_observed_data_and_indices_by_survey("my_survey", ClusterProperty.MASS)
