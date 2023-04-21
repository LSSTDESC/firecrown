"""
Tests for the TwoPoint module.
"""
import numpy as np

from firecrown.likelihood.gauss_family.statistic.two_point import _ell_for_xi


def test_ell_for_xi_no_rounding():
    res = _ell_for_xi(minimum=0, midpoint=5, maximum=80, n_log=5)
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    assert res.shape == expected.shape
    assert np.allclose(expected, res)


def test_ell_for_xi_doing_rounding():
    res = _ell_for_xi(minimum=1, midpoint=3, maximum=100, n_log=5)
    expected = np.array([1.0, 2.0, 3.0, 7.0, 17.0, 42.0, 100.0])
    assert np.allclose(expected, res)
