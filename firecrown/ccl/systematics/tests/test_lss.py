import pyccl as ccl
import numpy as np
from ..lss import LinearBiasSystematic, MagnificationBiasSystematic


class DummySource(object):
    pass


def test_linear_bias_systematic_smoke():
    src = DummySource()
    src.z_ = np.linspace(0, 2.0, 10)
    src.bias_ = 30.0
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67)
    gf = ccl.growth_factor(cosmo, 1.0 / (1.0 + src.z_))
    params = {
        '__alphag': 0.2,
        '__alphaz': 0.5,
        '__z_piv': 0.4}
    sys = LinearBiasSystematic(
        alphag='__alphag',
        alphaz='__alphaz',
        z_piv='__z_piv')

    sys.apply(cosmo, params, src)
    bias = 30.0 * gf**0.2 * ((1.0 + src.z_) / (1.0 + 0.4)) ** 0.5
    assert np.allclose(src.bias_, bias)


def test_magnification_bias_systematic_smoke():
    src = DummySource()
    src.z_ = np.linspace(0, 2.0, 10)
    src.mag_bias_ = 30.0
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67)
    params = {
        '__r_lim': 25.5,
        '__sig_c': 9.83,
        '__eta': 19.0,
        '__z_c': 0.39,
        '__z_m': 0.055}
    sys = MagnificationBiasSystematic(
        r_lim='__r_lim',
        sig_c='__sig_c',
        eta='__eta',
        z_c='__z_c',
        z_m='__z_m')

    sys.apply(cosmo, params, src)
    z_bar = 0.39 + 0.055 * (25.5 - 24)
    mag_bias = (
        30.0 / np.log(10) *
        (19.0 / 25.5 - 3 * 0.055 / z_bar + 1.5 * 0.055 *
         np.power(src.z_ / z_bar, 1.5) / z_bar))
    assert np.allclose(src.mag_bias_, mag_bias)
