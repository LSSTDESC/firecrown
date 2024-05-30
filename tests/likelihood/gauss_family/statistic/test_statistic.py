"""
Tests for the module firecrown.likelihood.gauss_family.statistic.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
import sacc

import firecrown.likelihood.gauss_family.statistic as stat
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap


VECTOR_CLASSES = (stat.TheoryVector, stat.DataVector)


def test_vector_create():
    vals = np.random.random_sample((10,))
    assert isinstance(vals, np.ndarray)
    assert vals.dtype == np.float64
    assert vals.shape == (10,)
    for cls in VECTOR_CLASSES:
        result = cls.create(vals)  # type: ignore
        assert isinstance(result, cls)
        assert result.shape == (10,)
        assert np.array_equal(vals, result)


def test_vector_from_list():
    vals = [1.5, 2.5, -3.0, 10.0]
    assert isinstance(vals, list)
    assert len(vals) == 4
    for cls in VECTOR_CLASSES:
        result = cls.from_list(vals)  # type: ignore
        assert isinstance(result, cls)
        assert result.shape == (4,)
        for i, val in enumerate(vals):
            assert result[i] == val


def test_vector_slicing():
    for cls in VECTOR_CLASSES:
        vec = cls.create(np.random.random_sample((12,)))  # type: ignore
        assert isinstance(vec, cls)
        middle_part = vec[3:6]
        assert middle_part.shape == (3,)
        assert isinstance(middle_part, cls)


def test_vector_copying():
    for cls in VECTOR_CLASSES:
        vec = cls.create(np.random.random_sample((12,)))  # type: ignore
        assert isinstance(vec, cls)
        vec_copy = vec.copy()
        assert vec_copy is not vec
        assert np.array_equal(vec, vec_copy)
        assert isinstance(vec_copy, cls)


def test_excplicit_vector_construction():
    for cls in VECTOR_CLASSES:
        vec = cls(shape=(4,), dtype=np.float64)
        assert isinstance(vec, cls)
        assert vec.shape == (4,)
        assert vec.dtype == np.float64


def test_ufunc_on_vector():
    data = np.array([0.0, 0.25, 0.50])
    expected = np.sin(data)
    for cls in VECTOR_CLASSES:
        vec = cls.create(data)  # type: ignore
        result = np.sin(vec)
        assert isinstance(result, cls)
        assert np.array_equal(result, expected)


def test_vector_residuals():
    theory = stat.TheoryVector.from_list([1.0, 2.0, 3.0])
    data = stat.DataVector.from_list([1.1, 2.1, 3.1])
    difference = stat.residuals(data, theory)
    assert isinstance(difference, np.ndarray)
    for cls in VECTOR_CLASSES:
        assert not isinstance(difference, cls)


def test_guarded_statistic_read_only_once(
    sacc_data_for_trivial_stat: sacc.Sacc, trivial_stats: list[stat.TrivialStatistic]
):
    gs = stat.GuardedStatistic(trivial_stats.pop())
    assert not gs.statistic.ready
    gs.read(sacc_data_for_trivial_stat)
    assert gs.statistic.ready
    with pytest.raises(
        RuntimeError, match="Firecrown has called read twice on a GuardedStatistic"
    ):
        gs.read(sacc_data_for_trivial_stat)


def test_guarded_statistic_get_data_before_read(trivial_stats):
    s = trivial_stats.pop()
    with pytest.raises(
        stat.StatisticUnreadError,
        match=f"The statistic {s} was used for "
        f"calculation before `read` was called.",
    ):
        g = stat.GuardedStatistic(s)
        _ = g.get_data_vector()


def test_statistic_get_data_vector_before_read():
    s = stat.TrivialStatistic()
    with pytest.raises(AssertionError):
        _ = s.get_data_vector()


def test_statistic_get_data_vector_after_read(sacc_data_for_trivial_stat):
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    assert_allclose(s.get_data_vector(), [1.0, 4.0, -3.0])


def test_statistic_get_theory_vector_before_compute():
    s = stat.TrivialStatistic()
    with pytest.raises(
        RuntimeError, match="The theory for statistic .* has not been computed yet\\."
    ):
        _ = s.get_theory_vector()


def test_statistic_get_theory_vector_after_compute(sacc_data_for_trivial_stat):
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    s.update(ParamsMap(mean=10.5))
    s.compute_theory_vector(ModelingTools())
    assert_allclose(s.get_theory_vector(), [10.5, 10.5, 10.5])


def test_statistic_get_theory_vector_after_reset(sacc_data_for_trivial_stat):
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    s.update(ParamsMap(mean=10.5))
    s.compute_theory_vector(ModelingTools())
    s.reset()
    with pytest.raises(
        RuntimeError, match="The theory for statistic .* has not been computed yet\\."
    ):
        _ = s.get_theory_vector()


def test_statistic_compute_theory_vector_before_read():
    s = stat.TrivialStatistic()
    with pytest.raises(
        RuntimeError,
        match="The statistic .* has not been updated with parameters\\.",
    ):
        s.compute_theory_vector(ModelingTools())


def test_statistic_compute_theory_vector_before_update(sacc_data_for_trivial_stat):
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    with pytest.raises(
        RuntimeError,
        match="The statistic .* has not been updated with parameters\\.",
    ):
        s.compute_theory_vector(ModelingTools())


def test_statistic_compute_theory_vector_after_update(sacc_data_for_trivial_stat):
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    s.update(ParamsMap(mean=10.5))

    assert_allclose(s.compute_theory_vector(ModelingTools()), [10.5, 10.5, 10.5])
