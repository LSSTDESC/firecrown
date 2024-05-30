"""Unit testsing for ConstGaussian
"""

import re

import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
from scipy.stats import chi2

import sacc

import firecrown.parameters
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.gauss_family import Statistic
from firecrown.likelihood.gauss_family.statistic import TrivialStatistic
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import (
    RequiredParameters,
    DerivedParameterCollection,
    SamplerParameter,
)


class StatisticWithoutIndices(TrivialStatistic):
    """This is a statistic that has no indices when read. It is only for
    testing.
    """

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the TrivialStatistic data, then nullify the sacc_indices."""
        super().read(sacc_data)
        self.sacc_indices = None


def test_require_nonempty_statistics():
    with pytest.raises(ValueError):
        _ = ConstGaussian(statistics=[])


def test_get_cov_fails_before_read(trivial_stats):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(AssertionError):
        _ = likelihood.get_cov()


def test_read_with_wrong_statistic_fails(sacc_data_for_trivial_stat):
    # Make the first statistic defective.
    defective_stat = StatisticWithoutIndices()
    likelihood = ConstGaussian(statistics=[defective_stat])
    with pytest.raises(
        RuntimeError,
        match="The statistic .* has no sacc_indices",
    ):
        likelihood.read(sacc_data_for_trivial_stat)


def test_get_cov_works_after_read(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    assert np.all(likelihood.get_cov() == np.diag([4.0, 9.0, 16.0]))


def test_get_cov_with_one_statistic(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    assert np.all(likelihood.get_cov(trivial_stats[0]) == np.diag([4.0, 9.0, 16.0]))


def test_get_cov_with_list_of_statistics(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    assert isinstance(trivial_stats, list)
    cov = likelihood.get_cov(trivial_stats)
    assert np.all(cov == np.diag([4.0, 9.0, 16.0]))


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


def test_compute_theory_vector_called_twice(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    res_1 = likelihood.compute_theory_vector(ModelingTools())
    res_2 = likelihood.compute_theory_vector(ModelingTools())
    assert np.all(res_1 == res_2)


def test_get_theory_vector_fails_before_read(trivial_stats):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "compute_theory_vector() must be called before get_theory_vector()"
        ),
    ):
        _ = likelihood.get_theory_vector()


def test_get_theory_vector_fails_before_update(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "compute_theory_vector() must be called before get_theory_vector()"
        ),
    ):
        _ = likelihood.get_theory_vector()


def test_get_theory_vector_fails_before_compute_theory_vector(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "compute_theory_vector() must be called before get_theory_vector()",
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
        match=re.escape(
            "compute_theory_vector() must be called before get_theory_vector()"
        ),
    ):
        _ = likelihood.get_theory_vector()


def test_compute_chisq_fails_before_read(trivial_stats):
    """Note that the error message from the direct call to compute_chisq notes
    that update() must be called; this can only be called after read()."""
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(
        AssertionError,
        match=re.escape("update() must be called before compute_chisq()"),
    ):
        _ = likelihood.compute_chisq(ModelingTools())


def test_compute_chisq(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
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
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return np.array([1.0, 4.0, -3.0]), np.array([1.0, 1.0, 1.0])

    likelihood.compute_theory_vector = compute_theory_vector  # type: ignore
    likelihood.compute = compute  # type: ignore

    assert likelihood.compute_chisq(ModelingTools()) == 2.0


def test_required_parameters(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    expected_params = RequiredParameters(
        params=[SamplerParameter(name="mean", default_value=0.0)]
    )
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


def test_make_realization_chisq(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    likelihood.compute_chisq(tools_with_vanilla_cosmology)

    new_sacc = likelihood.make_realization(sacc_data_for_trivial_stat)

    new_likelihood = ConstGaussian(statistics=[TrivialStatistic()])
    new_likelihood.read(new_sacc)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    new_likelihood.update(params)
    chisq = new_likelihood.compute_chisq(tools_with_vanilla_cosmology)

    # The new likelihood chisq is distributed as a chi-squared with 3 degrees of
    # freedom. We want to check that the new chisq is within the 1-10^-6 quantile
    # of the chi-squared distribution. This is equivalent to checking that the
    # new chisq is less than the 1-10^-6 quantile. This is expected to fail
    # 1 in 10^6 times.
    assert chisq < chi2.ppf(1.0 - 1.0e-6, df=3)


def test_make_realization_chisq_mean(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    likelihood.compute_chisq(tools_with_vanilla_cosmology)

    chisq_list = []
    for _ in range(1000):
        new_sacc = likelihood.make_realization(sacc_data_for_trivial_stat)

        new_likelihood = ConstGaussian(statistics=[TrivialStatistic()])
        new_likelihood.read(new_sacc)
        params = firecrown.parameters.ParamsMap(mean=10.5)
        new_likelihood.update(params)
        chisq = new_likelihood.compute_chisq(tools_with_vanilla_cosmology)
        chisq_list.append(chisq)

    # The new likelihood chisq is distributed as a chi-squared with 3 degrees of
    # freedom, so the mean is 3.0 and the variance is 6.0. Since we are computing
    # the mean of 1000 realizations, the variance of the mean is 6.0 / 1000.0.
    # We want to check that the new chisq is within 5 sigma of the mean.
    assert_allclose(np.mean(chisq_list), 3.0, atol=5.0 * np.sqrt(6.0 / 1000.0))


def test_make_realization_data_vector(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    likelihood.compute_chisq(tools_with_vanilla_cosmology)

    data_vector_list = []
    for _ in range(1000):
        new_sacc = likelihood.make_realization(sacc_data_for_trivial_stat)

        new_likelihood = ConstGaussian(statistics=[TrivialStatistic()])
        new_likelihood.read(new_sacc)
        params = firecrown.parameters.ParamsMap(mean=10.5)
        new_likelihood.update(params)
        data_vector = new_likelihood.get_data_vector()
        data_vector_list.append(data_vector)

    # The new likelihood data vector is distributed as a Gaussian with mean
    # equal to the theory vector and covariance equal to the covariance matrix.
    # We want to check that the new data vector is within 5 sigma of the mean.
    var_exact = np.array([4.0, 9.0, 16.0])
    assert_allclose(
        (np.mean(data_vector_list, axis=0) - np.array([10.5, 10.5, 10.5]))
        / np.sqrt(var_exact),
        0.0,
        atol=5.0 / np.sqrt(1000.0),
    )

    # The covariance can be computed as the covariance of the data vectors
    # minus the covariance of the theory vectors.
    covariance = np.cov(np.array(data_vector_list).T)
    assert_allclose(
        (covariance.diagonal() - var_exact) / np.sqrt(2.0 * var_exact**2 / 999.0),
        1.0,
        atol=5,
    )


def test_make_realization_no_noise(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    likelihood.compute_chisq(tools_with_vanilla_cosmology)

    new_sacc = likelihood.make_realization(sacc_data_for_trivial_stat, add_noise=False)

    new_likelihood = ConstGaussian(statistics=[TrivialStatistic()])
    new_likelihood.read(new_sacc)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    new_likelihood.update(params)

    assert_allclose(new_likelihood.get_data_vector(), likelihood.get_theory_vector())


def test_get_sacc_indices(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    idx = likelihood.get_sacc_indices()

    assert all(
        idx
        == np.concatenate(
            [stat.statistic.sacc_indices for stat in likelihood.statistics]
        )
    )


def test_get_sacc_indices_single_stat(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    idx = likelihood.get_sacc_indices(statistic=likelihood.statistics[0].statistic)

    assert all(idx == likelihood.statistics[0].statistic.sacc_indices)
