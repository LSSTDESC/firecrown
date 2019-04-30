import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import norm
from ..pz import PhotoZShiftBias, PhotoZSystematic


class DummySource(object):
    pass


def test_photoz_shift_smoke():
    z = np.linspace(0, 2, 500)
    dndz = np.exp(-0.5 * (z - 0.5)**2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, dndz)

    src = DummySource()
    src.z = z
    src.dndz = dndz
    src.z_ = z.copy()
    src.dndz_ = dndz.copy()
    src.dndz_interp = spline

    delta_z = 0.05
    params = {'delta_z': delta_z}

    sys = PhotoZShiftBias(delta_z='delta_z')
    sys.apply(None, params, src)

    mdndz_old = np.sum(z * dndz) / np.sum(dndz)
    mdndz_new = np.sum(src.z_ * src.dndz_) / np.sum(src.dndz_)
    assert np.abs(mdndz_new - mdndz_old - 0.05) < 1e-3
    assert np.allclose(src.z_, z)
    assert np.allclose(src.z, z)
    assert np.allclose(src.dndz, dndz)


def test_photoz_shift_fill_nan():
    z = np.linspace(0, 2, 500)
    dndz = np.exp(-0.5 * (z - 0.5)**2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, dndz)

    src = DummySource()
    src.z = z
    src.dndz = dndz
    src.z_ = z.copy()
    src.dndz_ = dndz.copy()
    src.dndz_interp = spline

    delta_z = 0.5
    params = {'delta_z': delta_z}

    sys = PhotoZShiftBias(delta_z='delta_z')
    sys.apply(None, params, src)

    z = np.linspace(0, 2, 500)
    dndz = np.exp(-0.5 * (z - 0.5)**2 / 0.25 / 0.25)
    spline = Akima1DInterpolator(z, dndz)

    msk = z - 0.5 <= 0
    assert np.all(src.dndz_[msk] == 0)
    assert np.all(src.dndz_[~msk] != 0)


def test_photoz_systematic_smoke():
    z = np.linspace(0, 1.5, 600)
    dndz = norm.pdf(z, 0.5, 0.1)

    src = DummySource()
    src.z = z
    src.dndz = dndz
    src.z_ = z.copy()
    src.dndz_ = dndz.copy()

    params = {
        'mu_0': 0.1,
        'mu_1': 0.0,
        'sigma': 0.05}
    sys = PhotoZSystematic(
        mu_0='mu_0',
        mu_1='mu_1',
        sigma='sigma')
    sys.apply(None, params, src)

    new_dndz = []
    h = z[1] - z[0]
    for z0 in z:
        jd = norm.pdf(z0, loc=z + 0.1, scale=0.05 * (1. + z)) * dndz
        new_dndz.append((sum(jd) - jd[-1]) * h)
    new_dndz = np.array(new_dndz)
    assert np.max(np.abs(src.dndz_ - new_dndz)) < 1e-3
    assert np.allclose(src.z_, z)
    assert np.allclose(src.z, z)
    assert np.allclose(src.dndz, dndz)
