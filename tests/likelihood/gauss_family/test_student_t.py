"""Unit testsing for Student-t distribution
"""

import pytest
import numpy as np

import sacc

import firecrown.parameters
from firecrown.likelihood.student_t import StudentT
from firecrown.likelihood.gauss_family import Statistic
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import (
    RequiredParameters,
    DerivedParameterCollection,
    ParamsMap,
    SamplerParameter,
)


@pytest.fixture(name="trivial_params_student_t")
def fixture_trivial_params_student_t(trivial_params) -> ParamsMap:
    """Return a ParamsMap with one parameter."""
    trivial_params["nu"] = 5.0
    return trivial_params


def test_require_nonempty_statistics():
    with pytest.raises(ValueError):
        _ = StudentT(statistics=[])


def test_update_fails_before_read(trivial_stats, trivial_params_student_t):
    likelihood = StudentT(statistics=trivial_stats)
    with pytest.raises(AssertionError):
        likelihood.update(trivial_params_student_t)


def test_get_cov_fails_before_read(trivial_stats):
    likelihood = StudentT(statistics=trivial_stats)
    with pytest.raises(AssertionError):
        _ = likelihood.get_cov()


def test_get_cov_works_after_read(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = StudentT(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    assert np.all(likelihood.get_cov() == np.diag([4.0, 9.0, 16.0]))


def test_chisquared(
    trivial_stats, sacc_data_for_trivial_stat, trivial_params_student_t
):
    trivial_params_student_t["nu"] = 5.0
    likelihood = StudentT(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params_student_t)
    assert likelihood.compute_chisq(ModelingTools()) == 2.0


def test_required_parameters(
    trivial_stats, sacc_data_for_trivial_stat, trivial_params_student_t
):
    likelihood = StudentT(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params_student_t)
    expected_params = RequiredParameters(
        params=[
            SamplerParameter(name="mean", default_value=0.0),
            SamplerParameter(name="nu"),
        ]
    )
    assert likelihood.required_parameters() == expected_params


def test_derived_parameters(
    trivial_stats, sacc_data_for_trivial_stat, trivial_params_student_t
):
    likelihood = StudentT(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params_student_t)
    expected_params = DerivedParameterCollection([])
    assert likelihood.get_derived_parameters() == expected_params


def test_reset(trivial_stats, sacc_data_for_trivial_stat, trivial_params_student_t):
    likelihood = StudentT(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params_student_t)
    assert not trivial_stats[0].computed_theory_vector
    assert likelihood.compute_loglike(ModelingTools()) == -1.013662770270411
    assert trivial_stats[0].computed_theory_vector
    likelihood.reset()
    assert not trivial_stats[0].computed_theory_vector


def test_missing_covariance(trivial_stats, sacc_with_data_points: sacc.Sacc):
    likelihood = StudentT(statistics=trivial_stats)
    with pytest.raises(
        RuntimeError,
        match="The StudentT likelihood requires a covariance, but the "
        "SACC data object being read does not have one.",
    ):
        likelihood.read(sacc_with_data_points)


def test_using_good_sacc(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = StudentT(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5, nu=5.0)
    likelihood.update(params)
    chisq = likelihood.compute_chisq(tools_with_vanilla_cosmology)
    assert isinstance(chisq, float)
    assert chisq > 0.0


def test_after_read_all_statistics_are_ready(
    trivial_stats, sacc_data_for_trivial_stat: sacc.Sacc
):
    likelihood = StudentT(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    for gs in likelihood.statistics:
        stat: Statistic = gs.statistic
        assert stat.ready
