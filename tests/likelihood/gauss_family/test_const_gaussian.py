"""Unit testsing for ConstGaussian
"""

import re
from typing import Tuple

import pytest
import numpy as np
import numpy.typing as npt

import sacc

import firecrown.parameters
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.gauss_family import Statistic
from firecrown.likelihood.gauss_family.statistic.statistic import TrivialStatistic
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import (
    RequiredParameters,
    DerivedParameterCollection,
)


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


def test_get_cov_with_statistics(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    assert np.all(likelihood.get_cov(trivial_stats[0]) == np.diag([4.0, 9.0, 16.0]))


def test_get_data_vector_fails_before_read(trivial_stats):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(
        AssertionError,
        match=re.escape("read() must be called before get_data_vector()"),
    ):
        _ = likelihood.get_data_vector()


def test_get_data_vector_works_after_read(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    assert np.all(likelihood.get_data_vector() == np.array([1.0, 4.0, -3.0]))


def test_compute_theory_vector_fails_before_read(trivial_stats):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(
        AssertionError,
        match=re.escape("update() must be called before compute_theory_vector()"),
    ):
        _ = likelihood.compute_theory_vector(ModelingTools())


def test_compute_theory_vector_fails_before_update(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    with pytest.raises(
        AssertionError,
        match=re.escape("update() must be called before compute_theory_vector()"),
    ):
        _ = likelihood.compute_theory_vector(ModelingTools())


def test_compute_theory_vector_works_after_read_and_update(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    assert np.all(
        likelihood.compute_theory_vector(ModelingTools())
        == np.array([10.5, 10.5, 10.5])
    )


def test_get_theory_vector_fails_before_read(trivial_stats):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(
        AssertionError,
        match=re.escape("update() must be called before get_theory_vector()"),
    ):
        _ = likelihood.get_theory_vector()


def test_get_theory_vector_fails_before_update(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    with pytest.raises(
        AssertionError,
        match=re.escape("update() must be called before get_theory_vector()"),
    ):
        _ = likelihood.get_theory_vector()


def test_get_theory_vector_fails_before_compute_theory_vector(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    with pytest.raises(
        RuntimeError,
        match=(
            "The theory vector has not been computed yet. "
            "Call compute_theory_vector first."
        ),
    ):
        _ = likelihood.get_theory_vector()


def test_get_theory_vector_works_after_read_update_and_compute_theory_vector(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    likelihood.compute_theory_vector(ModelingTools())
    assert np.all(likelihood.get_theory_vector() == np.array([10.5, 10.5, 10.5]))


def test_get_theory_vector_fails_after_read_update_compute_theory_vector_and_reset(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    likelihood.compute_theory_vector(ModelingTools())
    likelihood.reset()
    with pytest.raises(
        AssertionError,
        match=re.escape("update() must be called before get_theory_vector()"),
    ):
        _ = likelihood.get_theory_vector()


def test_chisquared(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    assert likelihood.compute_chisq(ModelingTools()) == 2.0


def test_deprecated_compute(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    with pytest.warns(DeprecationWarning):
        data_vector, theory_vector = likelihood.compute(ModelingTools())
    assert np.all(data_vector == np.array([1.0, 4.0, -3.0]))
    assert np.all(theory_vector == np.array([1.0, 1.0, 1.0]))


def test_chisquared_compute_vector_not_implemented(
    trivial_stats, sacc_data_for_trivial_stat, trivial_params
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)

    def compute_theory_vector(_tools: ModelingTools) -> npt.NDArray[np.float64]:
        raise NotImplementedError()

    def compute(
        _tools: ModelingTools,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return np.array([1.0, 4.0, -3.0]), np.array([1.0, 1.0, 1.0])

    likelihood.compute_theory_vector = compute_theory_vector  # type: ignore
    likelihood.compute = compute  # type: ignore

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
    assert likelihood.compute_loglike(ModelingTools()) == -1.0
    assert trivial_stats[0].computed_theory_vector
    likelihood.reset()
    assert not trivial_stats[0].computed_theory_vector


def test_missing_covariance(trivial_stats, sacc_with_data_points: sacc.Sacc):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(
        RuntimeError,
        match="The ConstGaussian likelihood requires a covariance, but the "
        "SACC data object being read does not have one.",
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


def test_write_to_sacc(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    likelihood.compute_chisq(tools_with_vanilla_cosmology)

    new_sacc = likelihood.write(sacc_data_for_trivial_stat)

    new_likelihood = ConstGaussian(statistics=[TrivialStatistic()])
    new_likelihood.read(new_sacc)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    new_likelihood.update(params)
    chisq = new_likelihood.compute_chisq(tools_with_vanilla_cosmology)

    assert np.isclose(chisq, 0.0)
