import os

import pandas as pd
import numpy as np

import pytest

import pyccl as ccl

from ..sources import NumberCountsSource
from ...systematics import MultiplicativeShearBias


@pytest.fixture(scope="session")
def lss_data(tmpdir_factory):
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


def test_lss_source_nosys(lss_data):
    src = NumberCountsSource(
        nz_data=lss_data['nz_data'],
        has_rsd=False,
        has_magnification=False,
        bias='blah2',
        scale=0.5)

    src.render(
        lss_data['cosmo'],
        lss_data['params'],
        lss_data['systematics_dict'])

    assert np.allclose(src.z_, lss_data['z'])
    assert np.allclose(src.nz_, lss_data['nz'])
    assert np.allclose(src.scale, 0.5)
    assert np.allclose(src.scale_, 0.5)
    assert src.systematics == []

    assert isinstance(src.tracer_, ccl.ClTracerNumberCounts)
    assert np.allclose(src.tracer_.z_n, lss_data['z'])
    assert np.allclose(src.tracer_.n, lss_data['nz'])
    assert np.allclose(src.tracer_.b, lss_data['params']['blah2'])


def test_lss_source_sys(lss_data):
    src = NumberCountsSource(
        nz_data=lss_data['nz_data'],
        has_rsd=False,
        has_magnification=False,
        bias='blah2',
        systematics=lss_data['systematics'])

    src.render(
        lss_data['cosmo'],
        lss_data['params'],
        lss_data['systematics_dict'])

    assert np.allclose(src.z_, lss_data['z'])
    assert np.allclose(src.nz_, lss_data['nz'])
    assert np.allclose(src.scale_, lss_data['scale_'])
    assert src.systematics == lss_data['systematics']

    assert isinstance(src.tracer_, ccl.ClTracerNumberCounts)
    assert np.allclose(src.tracer_.z_n, lss_data['z'])
    assert np.allclose(src.tracer_.n, lss_data['nz'])
    assert np.allclose(src.tracer_.b, lss_data['params']['blah2'])


def test_lss_source_mag(lss_data):
    src = NumberCountsSource(
        nz_data=lss_data['nz_data'],
        has_rsd=False,
        has_magnification=True,
        bias='blah2',
        mag_bias='blah6',
        systematics=lss_data['systematics'])

    src.render(
        lss_data['cosmo'],
        lss_data['params'],
        lss_data['systematics_dict'])

    assert np.allclose(src.z_, lss_data['z'])
    assert np.allclose(src.nz_, lss_data['nz'])
    assert np.allclose(src.scale_, lss_data['scale_'])
    assert src.systematics == lss_data['systematics']

    assert isinstance(src.tracer_, ccl.ClTracerNumberCounts)
    assert np.allclose(src.tracer_.z_n, lss_data['z'])
    assert np.allclose(src.tracer_.n, lss_data['nz'])
    assert np.allclose(src.tracer_.b, lss_data['params']['blah2'])
    assert np.allclose(src.tracer_.s, lss_data['params']['blah6'])
