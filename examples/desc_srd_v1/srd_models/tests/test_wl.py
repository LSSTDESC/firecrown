import numpy as np
import pyccl as ccl

from ..wl import (
    _mag_to_lum, _compute_red_frac_z_Az, KEBNLASystematic,
    DESCSRDv1MultiplicativeShearBias)

COSMO = ccl.Cosmology(
    Omega_b=0.0492,
    Omega_c=0.26639999999999997,  # = 0.3156 - 0.0492
    w0=-1.0,
    wa=0.0,
    h=0.6727,
    A_s=2.12655e-9,  # has sigma8 = 0.8310036
    n_s=0.9645)


class DummySource(object):
    pass


def test_mult_shear_bias():
    src = DummySource()
    src.z_ = np.linspace(0.0, 1.0, 100)
    src.dndz_ = src.z_**2 / 10 + src.z_
    src.scale_ = 1.0
    m = 0.05
    params = {'blah': m}

    nrm = np.sum(src.dndz_)
    fac = np.sum(src.dndz_ * (2 * src.z_ - 1.33) / 1.33) / nrm

    sys = DESCSRDv1MultiplicativeShearBias(m='blah')
    sys.apply(None, params, src)

    assert np.allclose(src.scale_, 1.0 + fac * 0.05)


def test_compute_red_frac_z_Az_redfrac():
    z = np.linspace(0.05, 3.55, 10)
    lpiv_beta_ia = _mag_to_lum(-22)
    beta_ia = 1.0
    vals = [_compute_red_frac_z_Az(
                _z, ccl.luminosity_distance(COSMO, 1 / (1.0 + _z)),
                beta_ia, lpiv_beta_ia)
            for _z in z]
    rf, _ = zip(*vals)
    rf = np.array(rf)

    # max red frac is always small for LSST depths
    assert np.max(rf) < 0.10

    # the red fraction is alwasy small at high redshift
    msk = z > 2.0
    assert np.all(rf[msk] < 0.01)


def test_ia_bias():
    src = DummySource()
    src.z_ = np.linspace(0.05, 3.55, 10)
    src.ia_bias_ = np.ones_like(src.z_)
    src.red_frac_ = np.ones_like(src.z_)

    eta_ia = 1.0
    eta_ia_highz = 4.0
    beta_ia = 1.0

    lpiv_beta_ia = _mag_to_lum(-22)
    vals = [_compute_red_frac_z_Az(
                _z, ccl.luminosity_distance(COSMO, 1 / (1.0 + _z)),
                beta_ia, lpiv_beta_ia)
            for _z in src.z_]
    rf, az = zip(*vals)

    params = {
        'a': eta_ia,
        'b': eta_ia_highz,
        'c': beta_ia}

    sys = KEBNLASystematic(
        eta_ia='a',
        eta_ia_highz='b',
        beta_ia='c')
    sys.apply(COSMO, params, src)

    # make sure it did something
    assert not np.allclose(src.ia_bias_, 1.0)

    # check the IA signal
    fac = np.power((1 + src.z_) / (1 + 0.3), eta_ia)
    fac *= (
        1.0 * (src.z_ < 0.7) +
        np.power((1 + src.z_) / (1 + 0.7), eta_ia_highz) * (src.z_ > 0.7))
    fac *= az
    fac *= rf
    assert np.allclose(src.ia_bias_, fac)
