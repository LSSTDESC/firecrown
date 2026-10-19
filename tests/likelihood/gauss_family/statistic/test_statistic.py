"""
Tests for the module firecrown.likelihood.statistic.
"""

import numpy as np
import numpy.typing as npt
import pytest
import sacc
from numpy.testing import assert_allclose

import firecrown.data_types
import firecrown.likelihood._statistic as stat
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import ParamsMap

VECTOR_CLASSES = (firecrown.data_types.TheoryVector, firecrown.data_types.DataVector)


@pytest.mark.parametrize("cls", VECTOR_CLASSES)
@pytest.mark.parametrize(
    "vals",
    [
        np.array([1, -2, 3], dtype=np.int16),
        np.array([1, 2, 3], dtype=np.uint64),
        np.array([1.25, -2.5, 3.75], dtype=np.float32),
    ],
)
def test_vector_converts_real_input(
    cls: (
        type[firecrown.data_types.DataVector] | type[firecrown.data_types.TheoryVector]
    ),
    vals: npt.NDArray[np.integer[npt.NBitBase] | np.floating[npt.NBitBase]],
) -> None:
    """Convert supported real values into nominal float64 vectors.

    :param cls: The public vector constructor.
    :param vals: Integer or lower-precision real observations or predictions.
    """
    result = cls.create(vals)
    assert type(result) is cls
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, vals)
    assert not np.shares_memory(result, vals)


def test_vector_create() -> None:
    vals = np.random.random_sample((10,))
    assert isinstance(vals, np.ndarray)
    assert vals.dtype == np.float64
    assert vals.shape == (10,)
    for cls in VECTOR_CLASSES:
        result = cls.create(vals)
        assert isinstance(result, cls)
        assert result.shape == (10,)
        assert np.array_equal(vals, result)


@pytest.mark.parametrize("cls", VECTOR_CLASSES)
@pytest.mark.parametrize(
    "vals",
    [
        np.array([True, False]),
        np.array([1 + 2j]),
        np.array(["1.25"]),
        np.array([b"1.25"]),
        np.array([1.25], dtype=object),
        np.array([(1.25,)], dtype=[("value", np.float64)]),
        np.array(["2026-10-18"], dtype="datetime64[D]"),
        np.array([1], dtype="timedelta64[D]"),
        np.ma.array([1.25], mask=[True]),
        np.ma.array([1.25], mask=False),
    ],
)
def test_vector_rejects_unsupported_input(
    cls: (
        type[firecrown.data_types.DataVector] | type[firecrown.data_types.TheoryVector]
    ),
    vals: npt.NDArray[np.generic],
) -> None:
    """Reject unsupported physical storage and masked values at construction.

    :param cls: The public vector constructor.
    :param vals: Deliberately unsupported physical data.
    """
    with pytest.raises(TypeError, match="real integer or floating|masked"):
        cls.create(vals)  # type: ignore[arg-type]  # Exercise invalid runtime ingress.


def test_vector_from_list() -> None:
    vals = [1.5, 2.5, -3.0, 10.0]
    assert isinstance(vals, list)
    assert len(vals) == 4
    for cls in VECTOR_CLASSES:
        result = cls.from_list(vals)
        assert isinstance(result, cls)
        assert result.shape == (4,)
        for i, val in enumerate(vals):
            assert result[i] == val


@pytest.mark.parametrize("cls", VECTOR_CLASSES)
def test_vector_sharing_and_empty_construction(
    cls: (
        type[firecrown.data_types.DataVector] | type[firecrown.data_types.TheoryVector]
    ),
) -> None:
    """Retain float64 sharing, list values, empty vectors, and shape allocation.

    :param cls: The public vector constructor.
    """
    for vals in (np.array([1.25, 2.5, 3.75]), np.arange(6, dtype=np.float64)[::2]):
        result = cls.create(vals)
        assert type(result) is cls
        assert result.dtype == np.float64
        assert np.shares_memory(result, vals)
        result[0] = -4.5
        assert vals[0] == -4.5
    for result in (
        cls.create(np.array([], dtype=np.int32)),
        cls.create(np.array([], dtype=np.float64)),
    ):
        assert type(result) is cls
        assert result.shape == (0,)
        assert result.dtype == np.float64
    lists: tuple[list[float], ...] = ([], [1, -2, 3])
    for values in lists:
        result = cls.from_list(values)
        assert type(result) is cls
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, values)
    allocated = cls(shape=(0,))
    assert type(allocated) is cls
    assert allocated.dtype == np.float64


@pytest.mark.parametrize("cls", VECTOR_CLASSES)
def test_vector_from_list_rejects_boolean(
    cls: (
        type[firecrown.data_types.DataVector] | type[firecrown.data_types.TheoryVector]
    ),
) -> None:
    """Reject Boolean physical values even through the list constructor.

    :param cls: The public vector constructor.
    """
    with pytest.raises(TypeError, match="real integer or floating"):
        cls.from_list([True, False])


def test_vector_slicing() -> None:
    for cls in VECTOR_CLASSES:
        vec = cls.create(np.random.random_sample((12,)))
        assert isinstance(vec, cls)
        middle_part = vec[3:6]
        assert middle_part.shape == (3,)
        assert isinstance(middle_part, cls)


def test_vector_copying() -> None:
    for cls in VECTOR_CLASSES:
        vec = cls.create(np.random.random_sample((12,)))
        assert isinstance(vec, cls)
        vec_copy = vec.copy()
        assert vec_copy is not vec
        assert np.array_equal(vec, vec_copy)
        assert isinstance(vec_copy, cls)


def test_excplicit_vector_construction() -> None:
    for cls in VECTOR_CLASSES:
        vec = cls(shape=(4,), dtype=np.float64)
        assert isinstance(vec, cls)
        assert vec.shape == (4,)
        assert vec.dtype == np.float64


def test_ufunc_on_vector() -> None:
    data = np.array([0.0, 0.25, 0.50])
    expected = np.sin(data)
    for cls in VECTOR_CLASSES:
        vec = cls.create(data)
        result = np.sin(vec)
        assert isinstance(result, cls)
        assert np.array_equal(result, expected)


def test_vector_residuals() -> None:
    theory = firecrown.data_types.TheoryVector.from_list([1.0, 2.0, 3.0])
    data = firecrown.data_types.DataVector.from_list([1.1, 2.1, 3.1])
    difference = firecrown.data_types.residuals(data, theory)
    assert isinstance(difference, np.ndarray)
    for cls in VECTOR_CLASSES:
        assert not isinstance(difference, cls)
    assert difference.dtype == np.float64
    assert_allclose(difference, [0.1, 0.1, 0.1], rtol=0, atol=1e-14)


def test_vector_residuals_convert_before_subtracting() -> None:
    """Avoid integer overflow and float32 subtraction before conversion."""
    data = firecrown.data_types.DataVector.create(np.array([127], dtype=np.int8))
    theory = firecrown.data_types.TheoryVector.create(np.array([-1], dtype=np.int8))
    result = firecrown.data_types.residuals(data, theory)
    assert type(result) is np.ndarray
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [128.0])
    data32 = firecrown.data_types.DataVector.create(np.array([1e8], dtype=np.float32))
    theory32 = firecrown.data_types.TheoryVector.create(np.array([1], dtype=np.float32))
    np.testing.assert_array_equal(
        firecrown.data_types.residuals(data32, theory32), [99999999.0]
    )
    empty = firecrown.data_types.residuals(
        firecrown.data_types.DataVector.from_list([]),
        firecrown.data_types.TheoryVector.from_list([]),
    )
    assert type(empty) is np.ndarray
    assert empty.dtype == np.float64
    assert empty.shape == (0,)


def test_guarded_statistic_read_only_once(
    sacc_data_for_trivial_stat: sacc.Sacc, trivial_stats: list[stat.TrivialStatistic]
) -> None:
    gs = stat.GuardedStatistic(trivial_stats.pop())
    assert not gs.statistic.ready
    gs.read(sacc_data_for_trivial_stat)
    assert gs.statistic.ready
    with pytest.raises(
        RuntimeError, match="Firecrown has called read twice on a GuardedStatistic"
    ):
        gs.read(sacc_data_for_trivial_stat)


def test_guarded_static_read_data_vector(
    sacc_data_for_trivial_stat: sacc.Sacc, trivial_stats: list[stat.TrivialStatistic]
) -> None:
    gs = stat.GuardedStatistic(trivial_stats.pop())
    gs.read(sacc_data_for_trivial_stat)
    assert_allclose(gs.get_data_vector(), [1.0, 4.0, -3.0])


def test_guarded_statistic_get_data_before_read(
    trivial_stats: list[stat.TrivialStatistic],
) -> None:
    s = trivial_stats.pop()
    with pytest.raises(
        stat.StatisticUnreadError,
        match=f"The statistic {s} was used for "
        f"calculation before `read` was called.",
    ):
        g = stat.GuardedStatistic(s)
        _ = g.get_data_vector()


def test_statistic_get_data_vector_before_read() -> None:
    s = stat.TrivialStatistic()
    with pytest.raises(AssertionError):
        _ = s.get_data_vector()


def test_guarded_statiistic_compute_theory_before_read() -> None:
    gs = stat.GuardedStatistic(stat.TrivialStatistic())
    with pytest.raises(
        RuntimeError,
        match="The statistic .* was used for calculation before `read` was called",
    ):
        gs.compute_theory_vector(ModelingTools())


def test_statistic_get_data_vector_after_read(
    sacc_data_for_trivial_stat: sacc.Sacc,
) -> None:
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    assert_allclose(s.get_data_vector(), [1.0, 4.0, -3.0])


def test_statistic_get_theory_vector_before_compute() -> None:
    s = stat.TrivialStatistic()
    with pytest.raises(
        RuntimeError, match="The theory for statistic .* has not been computed yet\\."
    ):
        _ = s.get_theory_vector()


def test_statistic_get_theory_vector_after_compute(
    sacc_data_for_trivial_stat: sacc.Sacc,
) -> None:
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    s.update(ParamsMap(mean=10.5))
    s.compute_theory_vector(ModelingTools())
    assert_allclose(s.get_theory_vector(), [10.5, 10.5, 10.5])


def test_statistic_get_theory_vector_after_reset(
    sacc_data_for_trivial_stat: sacc.Sacc,
) -> None:
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    s.update(ParamsMap(mean=10.5))
    s.compute_theory_vector(ModelingTools())
    s.reset()
    with pytest.raises(
        RuntimeError, match="The theory for statistic .* has not been computed yet\\."
    ):
        _ = s.get_theory_vector()


def test_statistic_compute_theory_vector_before_read() -> None:
    s = stat.TrivialStatistic()
    with pytest.raises(
        RuntimeError,
        match="The statistic .* has not been updated with parameters\\.",
    ):
        s.compute_theory_vector(ModelingTools())


def test_statistic_compute_theory_vector_before_update(
    sacc_data_for_trivial_stat: sacc.Sacc,
) -> None:
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    with pytest.raises(
        RuntimeError,
        match="The statistic .* has not been updated with parameters\\.",
    ):
        s.compute_theory_vector(ModelingTools())


def test_statistic_compute_theory_vector_after_update(
    sacc_data_for_trivial_stat: sacc.Sacc,
) -> None:
    s = stat.TrivialStatistic()
    s.read(sacc_data_for_trivial_stat)
    s.update(ParamsMap(mean=10.5))

    assert_allclose(s.compute_theory_vector(ModelingTools()), [10.5, 10.5, 10.5])
