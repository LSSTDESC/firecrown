import numpy as np
import firecrown.likelihood.gauss_family.statistic.statistic as stat


def test_vector_create():
    vals = np.random.random_sample((10,))
    assert isinstance(vals, np.ndarray)
    assert vals.dtype == np.float64
    assert vals.shape == (10,)
    for cls in (stat.TheoryVector, stat.DataVector):
        result = cls.create(vals)
        assert isinstance(result, cls)
        assert result.shape == (10,)
        assert np.array_equal(vals, result)


def test_vector_from_list():
    vals = [1.5, 2.5, -3.0, 10.0]
    assert isinstance(vals, list)
    assert len(vals) == 4
    for cls in (stat.TheoryVector, stat.DataVector):
        result = cls.from_list(vals)
        assert isinstance(result, cls)
        assert result.shape == (4,)
        for i, val in enumerate(vals):
            assert result[i] == val


def test_vector_slicing():
    for cls in (stat.TheoryVector, stat.DataVector):
        vec = cls.create(np.random.random_sample((12,)))
        assert isinstance(vec, cls)
        middle_part = vec[3:6]
        assert middle_part.shape == (3,)
        assert isinstance(middle_part, cls)


def test_vector_copying():
    for cls in (stat.TheoryVector, stat.DataVector):
        vec = cls.create(np.random.random_sample((12,)))
        assert isinstance(vec, cls)
        vec_copy = vec.copy()
        assert vec_copy is not vec
        assert np.array_equal(vec, vec_copy)
        assert isinstance(vec_copy, cls)
