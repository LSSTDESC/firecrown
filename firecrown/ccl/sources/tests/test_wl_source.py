import os

import pandas as pd
import numpy as np

import pytest

import pyccl as ccl

from ..sources import WLSource
from ...systematics import MultiplicativeShearBias


@pytest.fixture(scope="session")
def wl_data(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("data"))

    params = dict(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67)
    cosmo = ccl.Cosmology(**params)
    params['blah'] = 0.05
    params['blah2'] = 0.02
    params['blah6'] = 0.51
    wlm = MultiplicativeShearBias(m='blah')

    mn = 0.25
    z = np.linspace(0, 2, 50)
    nz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)

    df = pd.DataFrame({'z': z, 'nz': nz})
    nz_data = os.path.join(tmpdir, 'pz.csv')
    df.to_csv(nz_data, index=False)

    return {
        'cosmo': cosmo,
        'tmpdir': tmpdir,
        'params': params,
        'nz_data': nz_data,
        'systematics': ['wlm'],
        'systematics_dict': {'wlm': wlm},
        'scale_': 1.05,
        'z': z,
        'nz': nz}


def test_wl_source_nosys(wl_data):
    src = WLSource(
        nz_data=wl_data['nz_data'],
        has_intrinsic_alignment=False)

    src.render(
        wl_data['cosmo'],
        wl_data['params'],
        wl_data['systematics_dict'])

    assert np.allclose(src.z_, wl_data['z'])
    assert np.allclose(src.nz_, wl_data['nz'])
    assert np.allclose(src.scale_, 1.0)
    assert src.systematics == []

    assert isinstance(src.tracer_, ccl.ClTracerLensing)
    assert np.allclose(src.tracer_.z_n, wl_data['z'])
    assert np.allclose(src.tracer_.n, wl_data['nz'])


def test_wl_source_sys(wl_data):
    src = WLSource(
        nz_data=wl_data['nz_data'],
        has_intrinsic_alignment=False,
        systematics=wl_data['systematics'])

    src.render(
        wl_data['cosmo'],
        wl_data['params'],
        wl_data['systematics_dict'])

    assert np.allclose(src.z_, wl_data['z'])
    assert np.allclose(src.nz_, wl_data['nz'])
    assert np.allclose(src.scale_, wl_data['scale_'])
    assert src.systematics == wl_data['systematics']

    assert isinstance(src.tracer_, ccl.ClTracerLensing)
    assert np.allclose(src.tracer_.z_n, wl_data['z'])
    assert np.allclose(src.tracer_.n, wl_data['nz'])


def test_wl_source_with_ia(wl_data):
    src = WLSource(
        nz_data=wl_data['nz_data'],
        has_intrinsic_alignment=True,
        systematics=wl_data['systematics'],
        f_red='blah2',
        bias_ia='blah6')

    src.render(
        wl_data['cosmo'],
        wl_data['params'],
        wl_data['systematics_dict'])

    assert np.allclose(src.z_, wl_data['z'])
    assert np.allclose(src.nz_, wl_data['nz'])
    assert np.allclose(src.scale_, wl_data['scale_'])
    assert np.allclose(src.f_red_, wl_data['params']['blah2'])
    assert np.shape(src.f_red_) == np.shape(src.z_)
    assert np.allclose(src.bias_ia_, wl_data['params']['blah6'])
    assert np.shape(src.bias_ia_) == np.shape(src.z_)
    assert src.systematics == wl_data['systematics']

    assert isinstance(src.tracer_, ccl.ClTracerLensing)
    assert np.allclose(src.tracer_.z_n, wl_data['z'])
    assert np.allclose(src.tracer_.n, wl_data['nz'])
    assert np.allclose(src.tracer_.z_rf, wl_data['z'])
    assert np.allclose(src.tracer_.rf, wl_data['params']['blah2'])
    assert np.allclose(src.tracer_.z_ba, wl_data['z'])
    assert np.allclose(src.tracer_.ba, wl_data['params']['blah6'])
