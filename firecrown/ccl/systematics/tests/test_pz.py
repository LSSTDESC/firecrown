import numpy as np
from scipy.interpolate import Akima1DInterpolator
from ..pz import PhotoZShiftBias


class DummySource(object):
    pass


def test_photoz_shift_smoke():
    z = np.linspace(0, 2, 500)
    nz = np.exp(-0.5 * (z - 0.5)**2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, nz)

    src = DummySource()
    src.z = z
    src.nz = nz
    src.z_ = z.copy()
    src.nz_ = nz.copy()
    src.nz_interp = spline

    delta_z = 0.05
    params = {'delta_z': delta_z}

    sys = PhotoZShiftBias(delta_z='delta_z')
    sys.apply(None, params, src)

    mnz_old = np.sum(z * nz) / np.sum(nz)
    mnz_new = np.sum(src.z_ * src.nz_) / np.sum(src.nz_)
    assert np.abs(mnz_new - mnz_old - 0.05) < 1e-3
    assert np.allclose(src.z_, z)
    assert np.allclose(src.z, z)
    assert np.allclose(src.nz, nz)


def test_photoz_shift_fill_nan():
    z = np.linspace(0, 2, 500)
    nz = np.exp(-0.5 * (z - 0.5)**2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, nz)

    src = DummySource()
    src.z = z
    src.nz = nz
    src.z_ = z.copy()
    src.nz_ = nz.copy()
    src.nz_interp = spline

    delta_z = 0.5
    params = {'delta_z': delta_z}

    sys = PhotoZShiftBias(delta_z='delta_z')
    sys.apply(None, params, src)

    z = np.linspace(0, 2, 500)
    nz = np.exp(-0.5 * (z - 0.5)**2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, nz)

    msk = z - 0.5 <= 0
    assert np.all(src.nz_[msk] == 0)
    assert np.all(src.nz_[~msk] != 0)
