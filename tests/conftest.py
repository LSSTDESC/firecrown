"""
pytest configuration additions.
"""
from typing import final, Optional

import numpy as np
import pytest

import sacc

from firecrown.likelihood.gauss_family.statistic.statistic import (
    Statistic,
    DataVector,
    TheoryVector,
)
from firecrown import parameters
from firecrown.parameters import (
    RequiredParameters,
    DerivedParameterCollection,
    ParamsMap,
)

from firecrown.modeling_tools import ModelingTools


def pytest_addoption(parser):
    """Add handling of firecrown-specific options for the pytest test runner.

    --runslow: used to run tests marked as slow, which are otherwise not run.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Add new markers that can be set on pytest tests.

    Use the marker `slow` for any test that takes more than a second to run.
    Tests marked as `slow` are not run unless the user requests them by specifying
    the --runslow flag to the pytest program.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Apply our special markers and option handling for pytest."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# Fixtures


class TrivialStatistic(Statistic):
    """A minimal statistic for testing Gaussian likelihoods."""

    def __init__(self) -> None:
        """Initialize this statistic."""
        super().__init__()
        self.data_vector: Optional[DataVector] = None
        self.mean = parameters.create()
        self.computed_theory_vector = False

    def read(self, sacc_data: sacc.Sacc):
        """Read the necessary items from the sacc data."""

        our_data = sacc_data.get_mean(data_type="count")
        self.data_vector = DataVector.from_list(our_data)
        self.sacc_indices = np.arange(len(self.data_vector))
        super().read(sacc_data)

    @final
    def _reset(self):
        """Reset this statistic."""
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


@pytest.fixture(name="trivial_stats")
def make_stats():
    """Return a non-empty list of TrivialStatistics."""
    return [TrivialStatistic()]


@pytest.fixture(name="trivial_params")
def make_trivial_params() -> ParamsMap:
    """Return a ParamsMap with one parameter."""
    return ParamsMap({"mean": 1.0})


@pytest.fixture(name="sacc_data")
def make_sacc_data():
    """Create a trivial sacc.Sacc object."""
    result = sacc.Sacc()
    result.add_data_point("count", (), 1.0)
    result.add_data_point("count", (), 4.0)
    result.add_data_point("count", (), -3.0)
    result.add_covariance(np.diag([4.0, 9.0, 16.0]))
    return result
