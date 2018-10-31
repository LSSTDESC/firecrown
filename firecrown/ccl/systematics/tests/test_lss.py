import pyccl as ccl
import numpy as np
from ..lss import LinearBiasSystematic


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
