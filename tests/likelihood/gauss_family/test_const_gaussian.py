"""Unit testsing for ConstGaussian
"""
import pyccl
import pytest
import numpy as np

import sacc

import firecrown.parameters
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.gauss_family import Statistic
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import (
    RequiredParameters,
    DerivedParameterCollection,
    ParamsMap,
)


@pytest.fixture(name="sacc_with_data_points")
def fixture_sass_missing_covariance() -> sacc.Sacc:
    """Return a Sacc object for configuring a ConstGaussian likelihood,
    but which is missing a covariance matrix."""
    result = sacc.Sacc()
    result.add_tracer("misc", "sn_fake_sample")
    for cnt in [7.0, 4.0]:
        result.add_data_point("misc", ("sn_fake_sample",), cnt)
    return result


@pytest.fixture(name="sacc_with_covariance")
def fixture_sacc_with_covariance(sacc_with_data_points: sacc.Sacc) -> sacc.Sacc:
    result = sacc_with_data_points
    cov = np.array([[1.0, -0.5], [-0.5, 1.0]])
    result.add_covariance(cov)
    return result


@pytest.fixture(name="tools_with_vanilla_cosmology")
def fixture_tools_with_vanilla_cosmology():
    """Return a ModelingTools object containing the LCDM cosmology from
    pyccl."""
    result = ModelingTools()
    result.update(ParamsMap())
    result.prepare(pyccl.CosmologyVanillaLCDM())


def test_require_nonempty_statistics():
    with pytest.raises(ValueError):
        _ = ConstGaussian(statistics=[])


def test_get_cov_fails_before_read(trivial_stats):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(AssertionError):
        _ = likelihood.get_cov()


def test_get_cov_works_after_read(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    assert np.all(likelihood.get_cov() == np.diag([4.0, 9.0, 16.0]))


def test_chisquared(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    assert likelihood.compute_chisq(ModelingTools()) == 2.0


def test_required_parameters(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    expected_params = RequiredParameters(params_names=["mean"])
    assert likelihood.required_parameters() == expected_params


def test_derived_parameters(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    expected_params = DerivedParameterCollection([])
    assert likelihood.get_derived_parameters() == expected_params


def test_reset(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    assert not trivial_stats[0].computed_theory_vector
    assert likelihood.compute_chisq(ModelingTools()) == 2.0
    assert trivial_stats[0].computed_theory_vector
    likelihood.reset()
    assert not trivial_stats[0].computed_theory_vector


def test_missing_covariance(trivial_stats, sacc_with_data_points: sacc.Sacc):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(
        RuntimeError,
        match="The ConstGaussian likelihood "
        "requires a covariance, but the "
        "SACC data object being read does "
        "not have one.",
    ):
        likelihood.read(sacc_with_data_points)


def test_using_good_sacc(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    chisq = likelihood.compute_chisq(tools_with_vanilla_cosmology)
    assert isinstance(chisq, float)
    assert chisq > 0.0


def test_after_read_all_statistics_are_ready(
    trivial_stats, sacc_data_for_trivial_stat: sacc.Sacc
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    for gs in likelihood.statistics:
        stat: Statistic = gs.statistic
        assert stat.ready
