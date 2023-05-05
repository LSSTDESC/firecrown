"""
Tests for the firecrown.utils modle.
"""
from firecrown.utils import upper_triangle_indices


def test_upper_triangle_indices_nonzero():
    indices = list(upper_triangle_indices(3))
    assert indices == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_upper_triangle_indices_zero():
    indices = list(upper_triangle_indices(0))
    assert not indices
