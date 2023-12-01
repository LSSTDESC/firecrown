"""Tests for the cluster abundance data module."""
import pytest
import sacc
from firecrown.models.cluster.abundance_data import AbundanceData
from firecrown.models.cluster.properties import ClusterProperty


def test_create_abundance_data():
    s = sacc.Sacc()
    ad = AbundanceData(s, 2)

    assert ad.bin_dimensions == 2
    # pylint: disable=protected-access
    assert ad._mass_index == 2
    # pylint: disable=protected-access
    assert ad._redshift_index == 1
    # pylint: disable=protected-access
    assert ad._survey_index == 0


def test_get_survey_tracer_missing_survey_name(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data, 2)
    with pytest.raises(
        ValueError,
        match="The SACC file does not contain the SurveyTracer the_black_lodge.",
    ):
        ad.get_survey_tracer("the_black_lodge")


def test_get_survey_tracer_wrong_tracer_type(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data, 2)
    with pytest.raises(
        ValueError,
        match="The SACC tracer z_bin_tracer_1 is not a SurveyTracer.",
    ):
        ad.get_survey_tracer("z_bin_tracer_1")


def test_get_survey_tracer_returns_survey_tracer(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data, 2)
    survey = ad.get_survey_tracer("my_survey")
    assert survey is not None
    assert isinstance(survey, sacc.tracers.SurveyTracer)


def test_get_bin_edges_cluster_counts(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data, 2)
    bins = ad.get_bin_edges("my_survey", ClusterProperty.COUNTS)
    assert len(bins) == 2


def test_get_bin_edges_cluster_mass(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data, 2)
    bins = ad.get_bin_edges("my_survey", ClusterProperty.MASS)
    assert len(bins) == 2


def test_get_bin_edges_counts_and_mass(cluster_sacc_data: sacc.Sacc):
    ad = AbundanceData(cluster_sacc_data, 2)
    bins = ad.get_bin_edges(
        "my_survey", (ClusterProperty.MASS | ClusterProperty.COUNTS)
    )
    assert len(bins) == 2


def test_get_bin_edges_not_implemented_cluster_property_throws(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data, 2)
    with pytest.raises(NotImplementedError):
        ad.get_bin_edges("my_survey", ClusterProperty.SHEAR)


def test_observed_data_and_indices_by_survey_cluster_counts(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data, 2)
    data, indices = ad.get_observed_data_and_indices_by_survey(
        "my_survey", ClusterProperty.COUNTS
    )
    assert len(data) == 2
    assert len(indices) == 2


def test_observed_data_and_indices_by_survey_cluster_mass(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data, 2)
    data, indices = ad.get_observed_data_and_indices_by_survey(
        "my_survey", ClusterProperty.MASS
    )
    assert len(data) == 2
    assert len(indices) == 2


def test_observed_data_and_indices_by_survey_cluster_counts_and_mass(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data, 2)
    data, indices = ad.get_observed_data_and_indices_by_survey(
        "my_survey", ClusterProperty.MASS | ClusterProperty.COUNTS
    )
    assert len(data) == 4
    assert len(indices) == 4


def test_observed_data_and_indices_by_survey_not_implemented_throws(
    cluster_sacc_data: sacc.Sacc,
):
    ad = AbundanceData(cluster_sacc_data, 2)
    with pytest.raises(NotImplementedError):
        ad.get_observed_data_and_indices_by_survey("my_survey", ClusterProperty.SHEAR)
