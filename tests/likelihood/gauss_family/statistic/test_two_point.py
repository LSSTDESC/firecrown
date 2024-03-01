"""
Tests for the TwoPoint module.
"""

import numpy as np
import pytest

from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.gauss_family.statistic.source.number_counts import (
    NumberCounts,
)
from firecrown.likelihood.gauss_family.statistic.two_point import (
    _ell_for_xi,
    TwoPoint,
    TracerNames,
    TRACER_NAMES_TOTAL,
)


@pytest.fixture(name="source_0")
def fixture_source_0() -> NumberCounts:
    """Return an almost-default NumberCounts source."""
    return NumberCounts(sacc_tracer="lens_0")


@pytest.fixture(name="tools")
def fixture_tools() -> ModelingTools:
    """Return a trivial ModelingTools object."""
    return ModelingTools()


def test_ell_for_xi_no_rounding():
    res = _ell_for_xi(minimum=0, midpoint=5, maximum=80, n_log=5)
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    assert res.shape == expected.shape
    assert np.allclose(expected, res)


def test_ell_for_xi_doing_rounding():
    res = _ell_for_xi(minimum=1, midpoint=3, maximum=100, n_log=5)
    expected = np.array([1.0, 2.0, 3.0, 7.0, 17.0, 42.0, 100.0])
    assert np.allclose(expected, res)


def test_compute_theory_vector(source_0: NumberCounts):
    # To create the TwoPoint object we need at least one source.
    statistic = TwoPoint("galaxy_density_xi", source_0, source_0)
    assert isinstance(statistic, TwoPoint)

    # Before calling compute_theory_vector, we must get the TwoPoint object
    # into the correct state.
    # prediction = statistic.compute_theory_vector(tools)
    # assert isinstance(prediction, TheoryVector)


def test_tracer_names():
    assert TracerNames("", "") == TRACER_NAMES_TOTAL

    tn1 = TracerNames("cow", "pig")
    assert tn1[0] == "cow"
    assert tn1[1] == "pig"

    tn2 = TracerNames("cat", "dog")
    assert tn1 != tn2
    assert hash(tn1) != hash(tn2)
