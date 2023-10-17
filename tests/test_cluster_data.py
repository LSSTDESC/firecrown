import pytest
import numpy as np
from firecrown.models.cluster.abundance_data import AbundanceData
import sacc


@pytest.fixture
def complicated_sacc_data():
    cc = sacc.standard_types.cluster_counts
    mlm = sacc.standard_types.cluster_mean_log_mass

    s = sacc.Sacc()
    s.add_tracer("survey", "my_survey", 4000)
    s.add_tracer("survey", "not_my_survey", 4000)
    s.add_tracer("bin_z", "my_tracer1", 0, 2)
    s.add_tracer("bin_z", "my_tracer2", 2, 4)
    s.add_tracer("bin_richness", "my_other_tracer1", 0, 2)
    s.add_tracer("bin_richness", "my_other_tracer2", 2, 4)

    s.add_data_point(cc, ("my_survey", "my_tracer1", "my_other_tracer1"), 1)
    s.add_data_point(cc, ("my_survey", "my_tracer1", "my_other_tracer2"), 1)
    s.add_data_point(cc, ("not_my_survey", "my_tracer1", "my_other_tracer2"), 1)

    s.add_data_point(mlm, ("my_survey", "my_tracer1", "my_other_tracer1"), 1)
    s.add_data_point(mlm, ("my_survey", "my_tracer1", "my_other_tracer2"), 1)
    s.add_data_point(mlm, ("my_survey", "my_tracer2", "my_other_tracer2"), 1)
    s.add_data_point(mlm, ("my_survey", "my_tracer2", "my_other_tracer1"), 1)

    return s


def test_create_abundance_data_no_survey():
    with pytest.raises(
        ValueError, match="The SACC file does not contain the SurveyTracer"
    ):
        _ = AbundanceData(sacc.Sacc(), "survey", False, False)


def test_create_abundance_data_wrong_tracer():
    s = sacc.Sacc()
    s.add_tracer("bin_richness", "test", 0.1, 0.2)
    with pytest.raises(ValueError, match="The SACC tracer test is not a SurveyTracer"):
        _ = AbundanceData(s, "test", False, False)


def test_create_abundance_data():
    s = sacc.Sacc()
    s.add_tracer("survey", "mock_survey", 4000)
    ad = AbundanceData(s, "mock_survey", True, True)

    assert ad.cluster_counts is True
    assert ad.mean_log_mass is True
    assert ad.survey_nm == "mock_survey"
    assert ad.survey_tracer.sky_area == 4000
    assert ad._mass_index == 2
    assert ad._redshift_index == 1
    assert ad._survey_index == 0


def test_validate_tracers():
    s = sacc.Sacc()
    s.add_tracer("survey", "mock_survey", 4000)
    s.add_tracer("bin_z", "my_tracer", 0, 2)
    s.add_data_point(
        sacc.standard_types.cluster_counts, ("mock_survey", "my_tracer"), 1
    )
    ad = AbundanceData(s, "mock_survey", False, False)

    tracer_combs = np.array(
        s.get_tracer_combinations(sacc.standard_types.cluster_mean_log_mass)
    )
    with pytest.raises(
        ValueError,
        match="The SACC file does not contain any tracers for the"
        + f" {sacc.standard_types.cluster_mean_log_mass} data type",
    ):
        ad.validate_tracers(tracer_combs, sacc.standard_types.cluster_mean_log_mass)

    tracer_combs = np.array(
        s.get_tracer_combinations(sacc.standard_types.cluster_counts)
    )
    with pytest.raises(ValueError, match="The SACC file must contain 3 tracers"):
        ad.validate_tracers(tracer_combs, sacc.standard_types.cluster_counts)


def test_filtered_tracers(complicated_sacc_data):
    ad = AbundanceData(complicated_sacc_data, "my_survey", False, False)
    cc = sacc.standard_types.cluster_counts
    filtered_tracers, survey_mask = ad.get_filtered_tracers(cc)
    my_tracers = [
        ("my_survey", "my_tracer1", "my_other_tracer1"),
        ("my_survey", "my_tracer1", "my_other_tracer2"),
    ]
    assert (filtered_tracers == my_tracers).all()
    assert (survey_mask == [True, True, False]).all()


def test_get_data_and_indices(complicated_sacc_data):
    ad = AbundanceData(complicated_sacc_data, "my_survey", False, False)
    cc = sacc.standard_types.cluster_counts
    data, indices = ad.get_data_and_indices(cc)

    assert data == [1, 1]
    assert indices == [0, 1]


def test_get_bin_limits(complicated_sacc_data):
    ad = AbundanceData(complicated_sacc_data, "my_survey", False, False)
    cc = sacc.standard_types.cluster_counts
    limits = ad.get_bin_limits(cc)
    assert limits == [[(0, 2), (0, 2)], [(0, 2), (2, 4)]]
