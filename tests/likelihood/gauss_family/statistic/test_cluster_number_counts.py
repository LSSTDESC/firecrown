"""Tests for ClusterNumberCounts.
"""
import pytest

import sacc

from firecrown.likelihood.gauss_family.statistic.cluster_number_counts import (
    ClusterNumberCounts,
)


@pytest.fixture(name="minimal_stat")
def fixture_minimal_stat() -> ClusterNumberCounts:
    """Return a correctly initialized :python:`ClusterNumberCounts` object."""
    stat = ClusterNumberCounts(
        survey_tracer="SDSS",
        cluster_abundance=None,
        cluster_mass=None,
        cluster_redshift=None,
    )
    return stat


@pytest.fixture(name="missing_survey_tracer")
def fixture_missing_survey_tracer() -> sacc.Sacc:
    """Return a sacc.Sacc object that lacks a survey_tracer."""
    return sacc.Sacc()


@pytest.fixture(name="good_sacc_data")
def fixture_sacc_data():
    """Return a sacc.Sacc object sufficient to correctly set a
    :python:`ClusterNumberCounts` object.
    """
    data = sacc.Sacc()
    return data


def test_missing_survey_tracer(
    minimal_stat: ClusterNumberCounts, missing_survey_tracer: sacc.Sacc
):
    with pytest.raises(
        ValueError, match="The SACC file does not contain the SurveyTracer SDSS."
    ):
        minimal_stat.read(missing_survey_tracer)


def test_read_works(minimal_stat: ClusterNumberCounts, good_sacc_data: sacc.Sacc):
    """After read() is called, we should be able to get the statistic's

    :python:`DataVector` and also should be able to call
    :python:`compute_theory_vector`.
    """
