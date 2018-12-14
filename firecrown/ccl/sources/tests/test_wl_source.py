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
    dndz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)

    df = pd.DataFrame({'z': z, 'dndz': dndz})
    dndz_data = os.path.join(tmpdir, 'pz.csv')
    df.to_csv(dndz_data, index=False)

    return {
        'cosmo': cosmo,
        'tmpdir': tmpdir,
        'params': params,
        'dndz_data': dndz_data,
        'systematics': ['wlm'],
        'systematics_dict': {'wlm': wlm},
        'scale_': 1.05,
        'z': z,
        'dndz': dndz}


def test_wl_source_nosys(wl_data):
    src = WLSource(
        dndz_data=wl_data['dndz_data'],
        scale=0.5)

    src.render(
        wl_data['cosmo'],
        wl_data['params'],
        wl_data['systematics_dict'])

    assert np.allclose(src.z_, wl_data['z'])
    assert np.allclose(src.dndz_, wl_data['dndz'])
    assert np.allclose(src.scale, 0.5)
    assert np.allclose(src.scale_, 0.5)
    assert src.systematics == []

    assert isinstance(src.tracer_, ccl.WeakLensingTracer)
    assert np.allclose(src.tracer_.z_n, wl_data['z'])
    assert np.allclose(src.tracer_.n, wl_data['dndz'])


def test_wl_source_sys(wl_data):
    src = WLSource(
        dndz_data=wl_data['dndz_data'],
        systematics=wl_data['systematics'])

    src.render(
        wl_data['cosmo'],
        wl_data['params'],
        wl_data['systematics_dict'])

    assert np.allclose(src.z_, wl_data['z'])
    assert np.allclose(src.dndz_, wl_data['dndz'])
    assert np.allclose(src.scale_, wl_data['scale_'])
    assert src.systematics == wl_data['systematics']

    assert isinstance(src.tracer_, ccl.WeakLensingTracer)
    assert np.allclose(src.tracer_.z_n, wl_data['z'])
    assert np.allclose(src.tracer_.n, wl_data['dndz'])


def test_wl_source_with_ia(wl_data):
    src = WLSource(
        dndz_data=wl_data['dndz_data'],
        systematics=wl_data['systematics'],
        red_frac='blah2',
        ia_bias='blah6')

    src.render(
        wl_data['cosmo'],
        wl_data['params'],
        wl_data['systematics_dict'])

    assert np.allclose(src.z_, wl_data['z'])
    assert np.allclose(src.dndz_, wl_data['dndz'])
    assert np.allclose(src.scale_, wl_data['scale_'])
    assert np.allclose(src.red_frac_, wl_data['params']['blah2'])
    assert np.shape(src.red_frac_) == np.shape(src.z_)
    assert np.allclose(src.ia_bias_, wl_data['params']['blah6'])
    assert np.shape(src.ia_bias_) == np.shape(src.z_)
    assert src.systematics == wl_data['systematics']

    assert isinstance(src.tracer_, ccl.WeakLensingTracer)
    assert np.allclose(src.tracer_.z_n, wl_data['z'])
    assert np.allclose(src.tracer_.n, wl_data['dndz'])
    assert np.allclose(src.tracer_.z_rf, wl_data['z'])
    assert np.allclose(src.tracer_.rf, wl_data['params']['blah2'])
    assert np.allclose(src.tracer_.z_ba, wl_data['z'])
    assert np.allclose(src.tracer_.ba, wl_data['params']['blah6'])
