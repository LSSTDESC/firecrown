import pyccl

from ..cosmology import get_ccl_cosmology


def test_get_ccl_cosmology_smoke():
    params = dict(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67)
    cosmo = pyccl.Cosmology(**params)
    cosmo_get = get_ccl_cosmology(params)
    assert repr(cosmo) == repr(cosmo_get)


def test_get_ccl_cosmology_cached():
    params = dict(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67)
    cosmo1 = get_ccl_cosmology(params)
    cosmo2 = get_ccl_cosmology(params)
    assert cosmo1 is cosmo2
