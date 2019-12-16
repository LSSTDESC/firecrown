import numpy as np
import pytest

import sacc
import pyccl as ccl

from ..sources import NumberCountsSource
from ...systematics import MultiplicativeShearBias


@pytest.fixture
def lss_data():
    sacc_data = sacc.Sacc()

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
    sacc_data.add_tracer('NZ', 'trc1', z, dndz)

    # add extra data to make sure nothing weird is pulled back out
    mn = 0.5
    z = np.linspace(0, 2, 50)
    dndz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)
    sacc_data.add_tracer('NZ', 'trc2', z, dndz)

    return {
        'cosmo': cosmo,
        'sacc_tracer': 'trc1',
        'sacc_data': sacc_data,
        'params': params,
        'systematics': ['wlm'],
        'systematics_dict': {'wlm': wlm},
        'scale_': 1.05,
        'z': z,
        'dndz': dndz}


def test_lss_source_nosys(lss_data):
    src = NumberCountsSource(
        sacc_tracer=lss_data['sacc_tracer'],
        has_rsd=False,
        bias='blah2',
        scale=0.5)
    src.read(lss_data['sacc_data'])

    src.render(
        lss_data['cosmo'],
        lss_data['params'],
        lss_data['systematics_dict'])

    assert np.allclose(src.z_, lss_data['z'])
    assert np.allclose(src.dndz_, lss_data['dndz'])
    assert np.allclose(src.scale, 0.5)
    assert np.allclose(src.scale_, 0.5)
    assert src.systematics == []

    assert isinstance(src.tracer_, ccl.NumberCountsTracer)


def test_lss_source_sys(lss_data):
    src = NumberCountsSource(
        sacc_tracer=lss_data['sacc_tracer'],
        has_rsd=False,
        bias='blah2',
        systematics=lss_data['systematics'])
    src.read(lss_data['sacc_data'])

    src.render(
        lss_data['cosmo'],
        lss_data['params'],
        lss_data['systematics_dict'])

    assert np.allclose(src.z_, lss_data['z'])
    assert np.allclose(src.dndz_, lss_data['dndz'])
    assert np.allclose(src.scale_, lss_data['scale_'])
    assert src.systematics == lss_data['systematics']

    assert isinstance(src.tracer_, ccl.NumberCountsTracer)


def test_lss_source_mag(lss_data):
    src = NumberCountsSource(
        sacc_tracer=lss_data['sacc_tracer'],
        has_rsd=False,
        bias='blah2',
        mag_bias='blah6',
        systematics=lss_data['systematics'])
    src.read(lss_data['sacc_data'])

    src.render(
        lss_data['cosmo'],
        lss_data['params'],
        lss_data['systematics_dict'])

    assert np.allclose(src.z_, lss_data['z'])
    assert np.allclose(src.dndz_, lss_data['dndz'])
    assert np.allclose(src.scale_, lss_data['scale_'])
    assert src.systematics == lss_data['systematics']

    assert isinstance(src.tracer_, ccl.NumberCountsTracer)
