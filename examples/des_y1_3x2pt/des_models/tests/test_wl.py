import numpy as np
import pyccl as ccl

from ..wl import DESNLASystematic

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


def test_ia_bias():
    src = DummySource()
    src.z_ = np.linspace(0.05, 3.55, 10)
    src.ia_bias_ = np.ones_like(src.z_)

    eta_ia = 1.0
    Omega_b = 0.05
    Omega_c = 0.25

    params = {
        'a': eta_ia,
        'd': Omega_b,
        'e': Omega_c}

    sys = DESNLASystematic(
        eta_ia='a',
        Omega_b='d',
        Omega_c='e')
    sys.apply(COSMO, params, src)

    # make sure it did something
    assert not np.allclose(src.ia_bias_, 1.0)

    # check the IA signal
    fac = (
        (Omega_b + Omega_c) * 0.0134 /
        ccl.growth_factor(COSMO, 1.0 / (1.0 + src.z_)) *
        np.power((1 + src.z_) / (1 + 0.62), eta_ia))
    assert np.allclose(src.ia_bias_, fac)
