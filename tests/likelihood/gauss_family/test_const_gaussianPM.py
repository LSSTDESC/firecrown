"""Unit testing for ConstGaussianPM"""

import os
import re

import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
from scipy.stats import chi2

import sacc
import pyccl

import firecrown.parameters
from firecrown.likelihood import ConstGaussianPM, Statistic, TrivialStatistic, TwoPoint
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import (
    RequiredParameters,
    DerivedParameterCollection,
    SamplerParameter,
)
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood._gaussian_pointmass import PointMassData

from firecrown.metadata_types import TracerNames, Galaxies
from firecrown.metadata_functions import TwoPointHarmonicIndex
from firecrown.data_functions import extract_all_harmonic_data

# Tests that are inherited from ConstGaussian:


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
        _ = ConstGaussianPM(statistics=[])


def test_get_cov_fails_before_read(trivial_stats):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    with pytest.raises(AssertionError):
        _ = likelihood.get_cov()


def test_read_with_wrong_statistic_fails(sacc_data_for_trivial_stat):
    # Make the first statistic defective.
    defective_stat = StatisticWithoutIndices()
    likelihood = ConstGaussianPM(statistics=[defective_stat])
    with pytest.raises(
        RuntimeError,
        match="The statistic .* has no sacc_indices",
    ):
        likelihood.read(sacc_data_for_trivial_stat)


def test_get_cov_works_after_read(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    assert np.all(likelihood.get_cov() == np.diag([4.0, 9.0, 16.0]))


def test_get_cov_with_one_statistic(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    assert np.all(likelihood.get_cov(trivial_stats[0]) == np.diag([4.0, 9.0, 16.0]))


def test_get_cov_with_list_of_statistics(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    assert isinstance(trivial_stats, list)
    cov = likelihood.get_cov(trivial_stats)
    assert np.all(cov == np.diag([4.0, 9.0, 16.0]))


def test_get_data_vector_fails_before_read(trivial_stats):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    with pytest.raises(
        AssertionError,
        match=re.escape("read() must be called before get_data_vector()"),
    ):
        _ = likelihood.get_data_vector()


def test_get_data_vector_works_after_read(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    assert np.all(likelihood.get_data_vector() == np.array([1.0, 4.0, -3.0]))


def test_compute_theory_vector_fails_before_read(trivial_stats):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    with pytest.raises(
        AssertionError,
        match=re.escape("update() must be called before compute_theory_vector()"),
    ):
        _ = likelihood.compute_theory_vector(ModelingTools())


def test_compute_theory_vector_fails_before_update(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    with pytest.raises(
        AssertionError,
        match=re.escape("update() must be called before compute_theory_vector()"),
    ):
        _ = likelihood.compute_theory_vector(ModelingTools())


def test_compute_theory_vector_works_after_read_and_update(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    assert np.all(
        likelihood.compute_theory_vector(ModelingTools())
        == np.array([10.5, 10.5, 10.5])
    )


def test_compute_theory_vector_called_twice(trivial_stats, sacc_data_for_trivial_stat):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    res_1 = likelihood.compute_theory_vector(ModelingTools())
    res_2 = likelihood.compute_theory_vector(ModelingTools())
    assert np.all(res_1 == res_2)


def test_get_theory_vector_fails_before_read(trivial_stats):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
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
    likelihood = ConstGaussianPM(statistics=trivial_stats)
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
    likelihood = ConstGaussianPM(statistics=trivial_stats)
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
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(firecrown.parameters.ParamsMap(mean=10.5))
    likelihood.compute_theory_vector(ModelingTools())
    assert np.all(likelihood.get_theory_vector() == np.array([10.5, 10.5, 10.5]))


def test_get_theory_vector_fails_after_read_update_compute_theory_vector_and_reset(
    trivial_stats, sacc_data_for_trivial_stat
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
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
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    with (
        pytest.raises(
            AssertionError,
            match=re.escape("update() must be called before compute_chisq()"),
        ),
    ):
        _ = likelihood.compute_chisq(ModelingTools())


def test_compute_chisq(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
        assert likelihood.compute_chisq(ModelingTools()) == 2.0


def test_deprecated_compute(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    with pytest.warns(DeprecationWarning):
        data_vector, theory_vector = likelihood.compute(ModelingTools())
    assert np.all(data_vector == np.array([1.0, 4.0, -3.0]))
    assert np.all(theory_vector == np.array([1.0, 1.0, 1.0]))


def test_chisquared_compute_vector_not_implemented(
    trivial_stats, sacc_data_for_trivial_stat, trivial_params
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
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

    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
        assert likelihood.compute_chisq(ModelingTools()) == 2.0


def test_required_parameters(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    expected_params = RequiredParameters(
        params=[SamplerParameter(name="mean", default_value=0.0)]
    )
    assert likelihood.required_parameters() == expected_params


def test_derived_parameters(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    expected_params = DerivedParameterCollection([])
    assert likelihood.get_derived_parameters() == expected_params


def test_reset(trivial_stats, sacc_data_for_trivial_stat, trivial_params):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    likelihood.update(trivial_params)
    assert not trivial_stats[0].computed_theory_vector
    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
        assert likelihood.compute_loglike(ModelingTools()) == -1.0
    assert trivial_stats[0].computed_theory_vector
    likelihood.reset()
    assert not trivial_stats[0].computed_theory_vector


def test_missing_covariance(trivial_stats, sacc_with_data_points: sacc.Sacc):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    with pytest.raises(
        RuntimeError,
        match="The ConstGaussianPM likelihood requires a covariance, but the "
        "SACC data object being read does not have one.",
    ):
        likelihood.read(sacc_with_data_points)


def test_using_good_sacc(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
        chisq = likelihood.compute_chisq(tools_with_vanilla_cosmology)
    assert isinstance(chisq, float)
    assert chisq > 0.0


def test_after_read_all_statistics_are_ready(
    trivial_stats, sacc_data_for_trivial_stat: sacc.Sacc
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    for gs in likelihood.statistics:
        stat: Statistic = gs.statistic
        assert stat.ready


def test_make_realization_chisq(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
    tools_with_vanilla_cosmology: ModelingTools,
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
        likelihood.compute_chisq(tools_with_vanilla_cosmology)

    new_sacc = likelihood.make_realization(sacc_data_for_trivial_stat)

    new_likelihood = ConstGaussianPM(statistics=[TrivialStatistic()])
    new_likelihood.read(new_sacc)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    new_likelihood.update(params)
    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
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
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
        likelihood.compute_chisq(tools_with_vanilla_cosmology)

    chisq_list = []
    for _ in range(1000):
        new_sacc = likelihood.make_realization(sacc_data_for_trivial_stat)

        new_likelihood = ConstGaussianPM(statistics=[TrivialStatistic()])
        new_likelihood.read(new_sacc)
        params = firecrown.parameters.ParamsMap(mean=10.5)
        new_likelihood.update(params)
        with pytest.warns(
            UserWarning,
            match=re.escape("inverse covariance correction has not yet been computed."),
        ):
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
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
        likelihood.compute_chisq(tools_with_vanilla_cosmology)

    data_vector_list = []
    for _ in range(1000):
        new_sacc = likelihood.make_realization(sacc_data_for_trivial_stat)

        new_likelihood = ConstGaussianPM(statistics=[TrivialStatistic()])
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
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    likelihood.update(params)
    with pytest.warns(
        UserWarning,
        match=re.escape("inverse covariance correction has not yet been computed."),
    ):
        likelihood.compute_chisq(tools_with_vanilla_cosmology)

    new_sacc = likelihood.make_realization(sacc_data_for_trivial_stat, add_noise=False)

    new_likelihood = ConstGaussianPM(statistics=[TrivialStatistic()])
    new_likelihood.read(new_sacc)
    params = firecrown.parameters.ParamsMap(mean=10.5)
    new_likelihood.update(params)

    assert_allclose(new_likelihood.get_data_vector(), likelihood.get_theory_vector())


def test_get_sacc_indices(
    trivial_stats,
    sacc_data_for_trivial_stat: sacc.Sacc,
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
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
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    idx = likelihood.get_sacc_indices(statistic=likelihood.statistics[0].statistic)

    assert all(idx == likelihood.statistics[0].statistic.sacc_indices)


def test_access_required_parameters(
    trivial_stats,
):
    likelihood = ConstGaussianPM(statistics=trivial_stats)
    params = likelihood.required_parameters().get_default_values()
    assert params == {"mean": 0.0}


def test_create_ready(sacc_galaxy_cwindows, tp_factory):
    sacc_data, _, _ = sacc_galaxy_cwindows
    two_point_harmonics = extract_all_harmonic_data(sacc_data)

    two_points = TwoPoint.from_measurement(two_point_harmonics, tp_factory)
    size = np.sum([len(two_point.get_data_vector()) for two_point in two_points])

    likelihood = ConstGaussianPM.create_ready(two_points, np.diag(np.ones(size)))
    assert likelihood is not None
    assert isinstance(likelihood, ConstGaussianPM)


def test_create_ready_wrong_size(sacc_galaxy_cwindows, tp_factory):
    sacc_data, _, _ = sacc_galaxy_cwindows
    two_point_harmonics = extract_all_harmonic_data(sacc_data)

    two_points = TwoPoint.from_measurement(two_point_harmonics, tp_factory)
    size = np.sum([len(two_point.get_data_vector()) for two_point in two_points])

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The covariance matrix has shape (3, 3), "
            f"but the expected shape is at least ({size}, {size})."
        ),
    ):
        ConstGaussianPM.create_ready(two_points, np.diag([1.0, 2.0, 3.0]))


def test_create_ready_not_ready(tp_factory):
    metadata: TwoPointHarmonicIndex = {
        "data_type": "galaxy_density_xi",
        "tracer_names": TracerNames("lens0", "lens0"),
        "tracer_types": (Galaxies.COUNTS, Galaxies.COUNTS),
    }

    two_points = TwoPoint.from_metadata_index([metadata], tp_factory)

    with pytest.raises(
        RuntimeError,
        match="The statistic .* is not ready to be used.",
    ):
        ConstGaussianPM.create_ready(two_points, np.diag(np.ones(11)))


# Tests that are unique to ConstGaussianPM:


class MockStatistic:
    """A mock statistic with the relevant properties needed to test PM."""

    def __init__(self, sacc_data_type, source0, source1, thetas, get_data_vector):
        self.sacc_data_type = sacc_data_type
        self.source0 = source0
        self.source1 = source1
        self.thetas = thetas
        self.get_data_vector = get_data_vector


class MockSource:
    """A mock source with the relevant properties needed to test PM."""

    def __init__(self, sacc_tracer, tracer_args):
        self.sacc_tracer = sacc_tracer
        self.tracer_args = tracer_args


class MockTracerArgs:
    """A mock tracer with the relevant properties needed to test PM."""

    def __init__(self, z, dndz):
        self.z = z
        self.dndz = dndz


class MockStatisticContainer:
    """A mock statistic container with the relevant properties needed to test PM."""

    def __init__(self, statistic):
        self.statistic = statistic


@pytest.fixture(name="minimal_const_gaussian_PM")
def fixture_minimal_const_gaussian_PM() -> ConstGaussianPM:
    # Create minimal valid statistics for the class.
    z = np.array([0.1, 0.2, 0.3])
    dndz = np.array([1.0, 2.0, 3.0])
    tracer_args = MockTracerArgs(z, dndz)
    source0 = MockSource("lens0", tracer_args)
    source1 = MockSource("src0", tracer_args)
    statistic = MockStatistic(
        "galaxy_shearDensity_xi_t",
        source0,
        source1,
        [1.0, 2.0, 3.0],
        lambda: np.array([1.0, 2.0, 3.0]),
    )
    stat_container = MockStatisticContainer(statistic)
    likelihood = ConstGaussianPM(
        statistics=[
            TwoPoint(
                source0=nc.NumberCounts(sacc_tracer="lens0"),
                source1=wl.WeakLensing(sacc_tracer="src0"),
                sacc_data_type="galaxy_shearDensity_xi_t",
            )
        ]
    )
    # Replace statistics with mock - use object.__setattr__ to bypass type checking
    object.__setattr__(likelihood, "statistics", [stat_container])
    likelihood.cholesky = np.eye(3)
    likelihood.inv_cov = np.eye(3)
    # Use object.__setattr__ to replace final methods for testing
    object.__setattr__(
        likelihood, "get_theory_vector", lambda: np.array([1.0, 2.0, 3.0])
    )
    object.__setattr__(likelihood, "get_data_vector", lambda: np.array([1.0, 2.0, 3.0]))
    object.__setattr__(
        likelihood, "compute_theory_vector", lambda tools: np.array([1.0, 2.0, 3.0])
    )
    return likelihood


# pylint: disable=protected-access


def test_precomputed_warning(minimal_const_gaussian_PM):
    # Check that running the precomputation twice gives a warning.
    minimal_const_gaussian_PM._generate_maps()
    with pytest.warns(UserWarning):
        minimal_const_gaussian_PM._generate_maps()


def test_uneven_nz_size_error_lens(minimal_const_gaussian_PM):
    # Check the case of uneven lens N(z) lengths.
    tracer_args = MockTracerArgs(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]))
    tracer_args_short = MockTracerArgs(np.array([0.1, 0.2]), np.array([1.0, 2.0]))
    source0 = MockSource("lens", tracer_args_short)
    source1 = MockSource("src", tracer_args)
    statistic = MockStatistic(
        "galaxy_shearDensity_xi_t",
        source0,
        source1,
        [1.0, 2.0, 3.0],
        lambda: np.array([1.0, 2.0, 3.0]),
    )
    stat_container = MockStatisticContainer(statistic)
    minimal_const_gaussian_PM.statistics.append(stat_container)
    with pytest.raises(AssertionError):
        minimal_const_gaussian_PM._generate_maps()


def test_uneven_nz_size_error_source(minimal_const_gaussian_PM):
    # Check the case of uneven source N(z) lengths.
    tracer_args = MockTracerArgs(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]))
    tracer_args_short = MockTracerArgs(np.array([0.1, 0.2]), np.array([1.0, 2.0]))
    source0 = MockSource("lens", tracer_args)
    source1 = MockSource("src", tracer_args_short)
    statistic = MockStatistic(
        "galaxy_shearDensity_xi_t",
        source0,
        source1,
        [1.0, 2.0, 3.0],
        lambda: np.array([1.0, 2.0, 3.0]),
    )
    stat_container = MockStatisticContainer(statistic)
    minimal_const_gaussian_PM.statistics.append(stat_container)
    with pytest.raises(AssertionError):
        minimal_const_gaussian_PM._generate_maps()


def test_prepare_integrand(minimal_const_gaussian_PM):
    # Check that _prepare_integrand() behaves as expected.
    cosmo = pyccl.CosmologyVanillaLCDM()

    # Create mock PointMassData
    pm_data = PointMassData(
        theta=np.array([1.0, 2.0, 3.0]),
        row_lens_idx=np.array([0, 1, 2]),
        row_src_idx=np.array([0, 1, 2]),
        lens_tracers=["lens0"],
        src_tracers=["src0"],
        z_l=np.array([0.1, 0.2]),
        z_s=np.array([0.3, 0.4]),
        nzL_norm=np.array([[1.0, 2.0]]),
        nzS_norm=np.array([[1.0, 2.0]]),
    )
    minimal_const_gaussian_PM._pm_data = pm_data
    assert minimal_const_gaussian_PM._prepare_integrand(cosmo).shape == (2, 2)

    pm_data.z_l = np.array([0.0, 0.2])
    pm_data.z_s = np.array([0.0, 0.4])
    assert np.all(minimal_const_gaussian_PM._prepare_integrand(cosmo)) >= 0.0


def test_compute_betas(minimal_const_gaussian_PM):
    # Check that _compute_betas() generate the correct number of betas.
    cosmo = pyccl.CosmologyVanillaLCDM()
    tracer_args = MockTracerArgs(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]))
    source0 = MockSource("lens0", tracer_args)
    source1 = MockSource("src1", tracer_args)
    statistic = MockStatistic(
        "galaxy_shearDensity_xi_t",
        source0,
        source1,
        [1.0, 2.0, 3.0],
        lambda: np.array([1.0, 2.0, 3.0]),
    )
    stat_container = MockStatisticContainer(statistic)
    minimal_const_gaussian_PM.statistics.append(stat_container)
    minimal_const_gaussian_PM._generate_maps()
    assert np.shape(minimal_const_gaussian_PM._compute_betas(cosmo)) == (1, 2)


@pytest.fixture(name="sacc_data")
def fixture_sacc_data() -> sacc.Sacc:
    # Load sacc file
    saccfile = os.path.join(os.path.split(__file__)[0], "../../legacy_sacc_data.fits")
    return sacc.Sacc.load_fits(saccfile)


def test_PM_correction_matrix(sacc_data):
    # Check that the inverse covariance matrix changes.
    stats = {}
    stats["xip_src0_src0"] = TwoPoint(
        source0=wl.WeakLensing(sacc_tracer="src0"),
        source1=wl.WeakLensing(sacc_tracer="src0"),
        sacc_data_type="galaxy_shear_xi_plus",
    )
    stats["gammat_lens0_src0"] = TwoPoint(
        source0=wl.WeakLensing(sacc_tracer="src0"),
        source1=nc.NumberCounts(sacc_tracer="lens0"),
        sacc_data_type="galaxy_shearDensity_xi_t",
    )
    stats["wtheta_lens0_lens0"] = TwoPoint(
        source0=nc.NumberCounts(sacc_tracer="lens0"),
        source1=nc.NumberCounts(sacc_tracer="lens0"),
        sacc_data_type="galaxy_density_xi",
    )
    likelihood = ConstGaussianPM(statistics=list(stats.values()))
    likelihood.read(sacc_data)
    cosmo = pyccl.CosmologyVanillaLCDM()
    likelihood.compute_pointmass(cosmo)
    # Runtime checks to ensure inv_cov is not None for mypy
    assert likelihood.inv_cov is not None
    assert likelihood._pm_inv_cov_original is not None
    assert (likelihood.inv_cov != likelihood._pm_inv_cov_original).any()
    assert not np.isnan(likelihood.inv_cov).any()
    assert not np.isinf(likelihood.inv_cov).any()


def test_compute_chisq_with_correction(sacc_data):
    # Test that compute_chisq_impl works correctly when inv_cov_correction is set
    # This tests the truthy branch of the if statement at line 50
    stats = {}
    stats["gammat_lens0_src0"] = TwoPoint(
        source0=wl.WeakLensing(sacc_tracer="src0"),
        source1=nc.NumberCounts(sacc_tracer="lens0"),
        sacc_data_type="galaxy_shearDensity_xi_t",
    )
    likelihood = ConstGaussianPM(statistics=list(stats.values()))
    likelihood.read(sacc_data)
    cosmo = pyccl.CosmologyVanillaLCDM()

    # Compute point mass correction - this sets inv_cov_correction to a non-None value
    likelihood.compute_pointmass(cosmo)

    # Verify inv_cov_correction is set (not None)
    assert likelihood.inv_cov_correction is not None

    # Now call compute_chisq_impl directly with correct-sized residuals
    # to test the branch where inv_cov_correction is not None
    data_vector = likelihood.get_data_vector()
    residuals = np.zeros_like(data_vector)
    chisq = likelihood.compute_chisq_impl(residuals)

    # Verify it returns a finite value
    assert np.isfinite(chisq)


def test_get_lens_statistic_not_found(sacc_data):
    # Test that _get_lens_statistic raises StopIteration when lens tracer not found
    stats = {}
    stats["gammat_lens0_src0"] = TwoPoint(
        source0=wl.WeakLensing(sacc_tracer="src0"),
        source1=nc.NumberCounts(sacc_tracer="lens0"),
        sacc_data_type="galaxy_shearDensity_xi_t",
    )
    likelihood = ConstGaussianPM(statistics=list(stats.values()))
    likelihood.read(sacc_data)

    # Try to get a lens statistic that doesn't exist
    with pytest.raises(StopIteration, match="No lens statistic found for nonexistent"):
        likelihood._get_lens_statistic("nonexistent")


def test_get_src_statistic_not_found(sacc_data):
    # Test that _get_src_statistic raises StopIteration when source tracer not found
    stats = {}
    stats["gammat_lens0_src0"] = TwoPoint(
        source0=wl.WeakLensing(sacc_tracer="src0"),
        source1=nc.NumberCounts(sacc_tracer="lens0"),
        sacc_data_type="galaxy_shearDensity_xi_t",
    )
    likelihood = ConstGaussianPM(statistics=list(stats.values()))
    likelihood.read(sacc_data)

    # Try to get a source statistic that doesn't exist
    with pytest.raises(
        StopIteration, match="No source statistic found for nonexistent"
    ):
        likelihood._get_src_statistic("nonexistent")


def test_collect_data_vectors_missing_attributes():
    # Test that _collect_data_vectors raises StopIteration when statistics
    # are missing required attributes
    class IncompleteStatistic:
        def __init__(self):
            # Missing 'thetas' attribute
            self.sacc_data_type = "galaxy_shearDensity_xi_t"
            self.source0 = MockSource("lens0", None)
            self.source1 = MockSource("src0", None)
            # Missing self.thetas

    incomplete_stat = IncompleteStatistic()
    stat_container = MockStatisticContainer(incomplete_stat)

    # Create a real likelihood first, then replace statistics with mock
    likelihood = ConstGaussianPM(
        statistics=[
            TwoPoint(
                source0=nc.NumberCounts(sacc_tracer="lens0"),
                source1=wl.WeakLensing(sacc_tracer="src0"),
                sacc_data_type="galaxy_shearDensity_xi_t",
            )
        ]
    )
    # Replace statistics with mock - use object.__setattr__ to bypass type checking
    object.__setattr__(likelihood, "statistics", [stat_container])

    with pytest.raises(StopIteration, match="missing attributes"):
        likelihood._collect_data_vectors()


# pylint: enable=protected-access
