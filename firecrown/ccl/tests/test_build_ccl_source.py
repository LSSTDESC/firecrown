import numpy as np
from scipy.interpolate import Akima1DInterpolator

import pytest

import pyccl as ccl

from ..sources import build_ccl_source


@pytest.fixture
def source_keys():
    COSMO = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67)
    Z = np.linspace(0, 2, 500)
    NZ = np.exp(-0.5 * (Z - 0.5)**2 / 0.25 / 0.25)
    PZ_SPLINE = Akima1DInterpolator(Z, NZ)
    kwargs = {
        'cosmo': COSMO,
        'parameters': {},
        'kind': ccl.ClTracerLensing,
        'z_n': Z,
        'n': NZ,
        'pz_spline': PZ_SPLINE,
        'has_intrinsic_alignment': False,
        'systematics': None}
    return kwargs


def test_build_ccl_source_smoke(source_keys):
    tracer, scale = build_ccl_source(**source_keys)

    assert scale == 1.0
    assert isinstance(tracer, ccl.ClTracerLensing)
    assert np.allclose(tracer.z_n, source_keys['z_n'])
    assert np.allclose(tracer.n, source_keys['n'])


def test_build_ccl_source_sys_raises(source_keys):
    source_keys['systematics'] = {'ducks': {}}
    with pytest.raises(ValueError):
        build_ccl_source(**source_keys)


def test_build_ccl_source_kind_raises(source_keys):
    source_keys['kind'] = None
    with pytest.raises(ValueError):
        build_ccl_source(**source_keys)


def test_build_ccl_source_wl_sys(source_keys):
    source_keys['systematics'] = {
        'wl_mult_bias': {'m': 'cat'},
        'photoz_shift': {'delta_z': 'dog'}}
    source_keys['parameters'] = {'cat': 0.01, 'dog': 0.05}

    tracer, scale = build_ccl_source(**source_keys)

    assert scale == 1.01

    old_mnz = (
        np.sum(source_keys['z_n'] * source_keys['n']) /
        np.sum(source_keys['n']))
    new_mnz = (
        np.sum(tracer.z_n * tracer.n) /
        np.sum(tracer.n))
    assert np.abs(new_mnz - old_mnz - 0.05) < 1e-3
