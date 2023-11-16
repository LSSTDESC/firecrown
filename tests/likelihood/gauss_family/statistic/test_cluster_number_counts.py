"""Tests for the ClusterNumberCounts statistic.
"""
import pytest

import pyccl.halos
import sacc

from firecrown.likelihood.gauss_family.statistic.cluster_number_counts import (
    ClusterNumberCounts,
    ClusterAbundance,
)
from firecrown.models.cluster_mass_rich_proxy import ClusterMassRich
from firecrown.models.cluster_redshift_spec import ClusterRedshiftSpec


@pytest.fixture(name="minimal_stat")
def fixture_minimal_stat() -> ClusterNumberCounts:
    """Return a correctly initialized :class:`ClusterNumberCounts` object."""
    stat = ClusterNumberCounts(
        survey_tracer="SDSS",
        cluster_abundance=ClusterAbundance(
            halo_mass_definition=pyccl.halos.MassDef(0.5, "matter"),
            halo_mass_function_name="200m",
            halo_mass_function_args={},
        ),
        cluster_mass=ClusterMassRich(pivot_mass=10.0, pivot_redshift=1.25),
        cluster_redshift=ClusterRedshiftSpec(),
    )
    return stat


@pytest.fixture(name="missing_survey_tracer")
def fixture_missing_survey_tracer() -> sacc.Sacc:
    """Return a sacc.Sacc object that lacks a survey_tracer."""
    return sacc.Sacc()


@pytest.fixture(name="good_sacc_data")
def fixture_sacc_data():
    """Return a sacc.Sacc object sufficient to correctly set a
    :class:`ClusterNumberCounts` object.
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


def test_read_works():
    """After read() is called, we should be able to get the statistic's

    :class:`DataVector` and also should be able to call
    :meth:`compute_theory_vector`.
    """
