import numpy as np

from ..kecorrections import kcorr, ecorr


def test_kcorr():
    # test extrapolation doesn't error
    assert np.allclose(kcorr(0.0), 0.021)
    assert np.allclose(kcorr(4), 6.888)

    # try a value at random in the middle
    assert np.allclose(kcorr(0.76), 1.763)


def test_ecorr():
    # test extrapolation doesn't error
    assert np.allclose(ecorr(0.0), -0.024)
    assert np.allclose(ecorr(4), -8.549)

    # try a value at random in the middle
    assert np.allclose(ecorr(0.76), -0.960)
