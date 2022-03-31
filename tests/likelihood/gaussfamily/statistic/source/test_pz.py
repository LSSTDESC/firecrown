import numpy as np
from scipy.interpolate import Akima1DInterpolator
from ..pz import PhotoZShiftBias


class DummySource(object):
    pass


def test_photoz_shift_smoke():
    z = np.linspace(0, 2, 500)
    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, dndz)

    src = DummySource()
    src.z = z
    src.dndz = dndz
    src.z_ = z.copy()
    src.dndz_ = dndz.copy()
    src.dndz_interp = spline

    delta_z = 0.05
    params = {"delta_z": delta_z}

    sys = PhotoZShiftBias(delta_z="delta_z")
    sys.apply(None, params, src)

    mdndz_old = np.sum(z * dndz) / np.sum(dndz)
    mdndz_new = np.sum(src.z_ * src.dndz_) / np.sum(src.dndz_)
    assert np.abs(mdndz_new - mdndz_old - 0.05) < 1e-3
    assert np.allclose(src.z_, z)
    assert np.allclose(src.z, z)
    assert np.allclose(src.dndz, dndz)


def test_photoz_shift_fill_nan():
    z = np.linspace(0, 2, 500)
    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, dndz)

    src = DummySource()
    src.z = z
    src.dndz = dndz
    src.z_ = z.copy()
    src.dndz_ = dndz.copy()
    src.dndz_interp = spline

    delta_z = 0.5
    params = {"delta_z": delta_z}

    sys = PhotoZShiftBias(delta_z="delta_z")
    sys.apply(None, params, src)

    z = np.linspace(0, 2, 500)
    dndz = np.exp(-0.5 * (z - 0.5) ** 2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, dndz)

    msk = z - 0.5 <= 0
    assert np.all(src.dndz_[msk] == 0)
    assert np.all(src.dndz_[~msk] != 0)
