import pyccl as ccl
import numpy as np
from ..wl import MultiplicativeShearBias, LinearAlignmentSystematic


class DummySource(object):
    pass


def test_mult_shear_bias_smoke():
    src = DummySource()
    src.scale_ = 1.0
    m = 0.05
    params = {'blah': m}

    sys = MultiplicativeShearBias(m='blah')
    sys.apply(None, params, src)

    assert np.allclose(src.scale_, 1.05)


def test_linear_alignment_systematic_smoke():
    src = DummySource()
    src.z_ = np.linspace(0, 2.0, 10)
    src.ia_bias_ = 30.0
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
        '__z_piv': 0.4,
        '__Omega_b': 0.045,
        '__Omega_c': 0.27}
    sys = LinearAlignmentSystematic(
        alphag='__alphag',
        alphaz='__alphaz',
        z_piv='__z_piv',
        Omega_b='__Omega_b',
        Omega_c='__Omega_c')

    sys.apply(cosmo, params, src)
    bias_ia = (
        (0.045 + 0.27) * 0.0134 *
        30.0 * gf**0.2 * ((1.0 + src.z_) / (1.0 + 0.4)) ** 0.5)
    assert np.allclose(src.ia_bias_, bias_ia)
