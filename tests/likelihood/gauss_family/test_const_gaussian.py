"""Unit testsing for ConstGaussian
"""

from typing import final, Optional
import pytest
import numpy as np
import sacc

from firecrown.likelihood.gauss_family.statistic.statistic import (
    Statistic,
    DataVector,
    TheoryVector,
)
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown import parameters
from firecrown.parameters import (
    RequiredParameters,
    DerivedParameterCollection,
    ParamsMap,
)


class TrivialStatistic(Statistic):
    """A minimal statistic for testing Gaussian likelihoods."""

    def __init__(self) -> None:
        """Initialize this statistic."""
        super().__init__()
        self.data_vector: Optional[DataVector] = None
        self.mean = parameters.create()
        self.computed_theory_vector = False

    def read(self, sacc_data: sacc.Sacc):
        """This trivial class does not actually need to read anything."""

        our_data = sacc_data.get_mean(data_type="count")
        self.data_vector = DataVector.from_list(our_data)
        self.sacc_indices = np.arange(len(self.data_vector))

    @final
    def _reset(self):
        """Reset this statistic. This implementation has nothing to do."""
        self.computed_theory_vector = False

    @final
    def _required_parameters(self) -> RequiredParameters:
        """Return an empty RequiredParameters."""
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Return an empty DerivedParameterCollection."""
        return DerivedParameterCollection([])

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none."""
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, _: ModelingTools) -> TheoryVector:
        """Return a fixed theory vector."""
        self.computed_theory_vector = True
        return TheoryVector.from_list([self.mean, self.mean, self.mean])


@pytest.fixture(name="stats")
def make_stats():
    """Return a non-empty list of TrivialStatistics."""
    return [TrivialStatistic()]


@pytest.fixture(name="sacc_data")
def make_sacc_data():
    """Create a trivial sacc.Sacc object."""
    result = sacc.Sacc()
    result.add_data_point("count", (), 1.0)
    result.add_data_point("count", (), 4.0)
    result.add_data_point("count", (), -3.0)
    result.add_covariance(np.diag([4.0, 9.0, 16.0]))
    return result


@pytest.fixture(name="trivial_params")
def make_trivial_params() -> ParamsMap:
    """Return a ParamsMap with one parameter."""
    return ParamsMap({"mean": 1.0})


def test_require_nonempty_statistics():
    with pytest.raises(ValueError):
        _ = ConstGaussian(statistics=[])


def test_get_cov_fails_before_read(stats):
    likelihood = ConstGaussian(statistics=stats)
    with pytest.raises(AssertionError):
        _ = likelihood.get_cov()


def test_get_cov_works_after_read(stats, sacc_data):
    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)
    assert np.all(likelihood.get_cov() == np.diag([4.0, 9.0, 16.0]))


def test_chisquared(stats, sacc_data, trivial_params):
    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)
    likelihood.update(trivial_params)
    assert likelihood.compute_chisq(ModelingTools()) == 2.0


def test_required_parameters(stats, sacc_data, trivial_params):
    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)
    likelihood.update(trivial_params)
    expected_params = RequiredParameters(params_names=["mean"])
    assert likelihood.required_parameters() == expected_params


def test_derived_parameters(stats, sacc_data, trivial_params):
    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)
    likelihood.update(trivial_params)
    expected_params = DerivedParameterCollection([])
    assert likelihood.get_derived_parameters() == expected_params


def test_reset(stats, sacc_data, trivial_params):
    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)
    likelihood.update(trivial_params)
    assert not stats[0].computed_theory_vector
    assert likelihood.compute_chisq(ModelingTools()) == 2.0
    assert stats[0].computed_theory_vector
    likelihood.reset()
    assert not stats[0].computed_theory_vector
