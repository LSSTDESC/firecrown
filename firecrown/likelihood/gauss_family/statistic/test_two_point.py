import pytest
import numpy as np

from .two_point import _ell_for_xi


def test_ell_for_xi():
    res = _ell_for_xi(min=0.0, mid=10.0, max=80.0, n_log=4)
    assert res == np.array([0.0, 10.0, 20.0, 40.0, 80.0])
