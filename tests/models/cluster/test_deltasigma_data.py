"""Tests for the cluster delta sigma data module."""

import pytest
import sacc
from firecrown.models.cluster import (
    ClusterProperty,
    DeltaSigmaData,
)


def test_create_deltasigma_data():
    s = sacc.Sacc()
    dsd = DeltaSigmaData(s)

    # pylint: disable=protected-access
    assert dsd._mass_index == 2
    # pylint: disable=protected-access
    assert dsd._redshift_index == 1
    # pylint: disable=protected-access
    assert dsd._survey_index == 0
    # pylint: disable=protected-access
    assert dsd._radius_index == 3


def test_get_survey_tracer_missing_survey_name(cluster_sacc_data: sacc.Sacc):
    dsd = DeltaSigmaData(cluster_sacc_data)
    with pytest.raises(
        ValueError,
        match="The SACC file does not contain the SurveyTracer the_black_lodge.",
    ):
        dsd.get_survey_tracer("the_black_lodge")


def test_get_survey_tracer_wrong_tracer_type(cluster_sacc_data: sacc.Sacc):
    dsd = DeltaSigmaData(cluster_sacc_data)
    with pytest.raises(
        ValueError,
        match="The SACC tracer z_bin_tracer_1 is not a SurveyTracer.",
    ):
        dsd.get_survey_tracer("z_bin_tracer_1")


def test_wrong_property(cluster_sacc_data: sacc.Sacc) -> None:
    dsd = DeltaSigmaData(cluster_sacc_data)
    with pytest.raises(
        ValueError,
        match=f"The property must be {ClusterProperty.DELTASIGMA}.",
    ):
        dsd.get_bin_edges("my_survey", ClusterProperty.COUNTS)


def test_wrong_tracer_number(cluster_sacc_data: sacc.Sacc) -> None:
    dsd = DeltaSigmaData(cluster_sacc_data)
    # pylint: disable=no-member
    cs = sacc.standard_types.cluster_shear
    tracers_n = 3
    with pytest.raises(
        ValueError,
        match=f"The SACC file must contain {tracers_n} tracers for the "
        f"{cs} data type.",
    ):
        # pylint: disable=protected-access
        dsd._all_bin_combinations_for_data_type(cs, tracers_n)


def test_get_survey_tracer_returns_survey_tracer(cluster_sacc_data: sacc.Sacc):
    dsd = DeltaSigmaData(cluster_sacc_data)
    survey = dsd.get_survey_tracer("my_survey")
    assert survey is not None
    assert isinstance(survey, sacc.tracers.SurveyTracer)


def test_get_bin_edges_cluster_radius(cluster_sacc_data: sacc.Sacc):
    dsd = DeltaSigmaData(cluster_sacc_data)
    bins = dsd.get_bin_edges("my_survey", ClusterProperty.DELTASIGMA)
    assert len(bins) == 2


def test_observed_data_and_indices_by_survey_cluster_deltasigma(
    cluster_sacc_data: sacc.Sacc,
):
    dsd = DeltaSigmaData(cluster_sacc_data)
    data, indices = dsd.get_observed_data_and_indices_by_survey(
        "my_survey", ClusterProperty.DELTASIGMA
    )
    assert len(data) == 2
    assert len(indices) == 2


def test_observed_data_and_indices_wrong_property():
    """Test error when wrong property used with DeltaSigmaData."""
    s = sacc.Sacc()
    ads = DeltaSigmaData(s)

    # pylint: disable=no-member
    with pytest.raises(
        ValueError,
        match=f"The property should be related to the "
        f"{sacc.standard_types.cluster_shear} data type.",
    ):
        ads.get_observed_data_and_indices_by_survey("my_survey", ClusterProperty.MASS)
