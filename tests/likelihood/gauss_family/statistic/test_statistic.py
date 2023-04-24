"""
Tests for the module firecrown.likelihood.gauss_family.statistic.statistic.
"""
import numpy as np
import firecrown.likelihood.gauss_family.statistic.statistic as stat

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
