import pytest
import numpy as np

from .two_point import _ell_for_xi


def test_ell_for_xi():
    res = _ell_for_xi(min=0, mid=5, max=80, n_log=5)
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    assert np.all(expected == res)

    res = _ell_for_xi(min=0, mid=2, max=50, n_log=3)
    expected = np.array([0.0, 1.0, 2.0, 10.0, 50.0])
    assert np.all(expected == res)

    res = _ell_for_xi(min=1, mid=3, max=100, n_log=5)
    expected = np.array([1., 2., 3., 7., 17., 42., 100.])
    assert np.all(expected == res)