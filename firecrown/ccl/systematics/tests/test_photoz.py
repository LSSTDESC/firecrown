import numpy as np
from scipy.interpolate import Akima1DInterpolator
from .._photoz import photoz_shift


def test_photoz_shift_smoke():
    z = np.linspace(0, 2, 500)
    nz = np.exp(-0.5 * (z - 0.5)**2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, nz)

    new_nz = photoz_shift(z, spline, 0.05)

    mnz_old = np.sum(z * nz) / np.sum(nz)
    mnz_new = np.sum(z * new_nz) / np.sum(new_nz)
    assert np.abs(mnz_new - mnz_old - 0.05) < 1e-3


def test_photoz_shift_fill_nan():
    z = np.linspace(0, 2, 500)
    nz = np.exp(-0.5 * (z - 0.5)**2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, nz)

    new_nz = photoz_shift(z, spline, 0.5)
    msk = z - 0.5 <= 0
    assert np.all(new_nz[msk] == 0)
    assert np.all(new_nz[~msk] != 0)
