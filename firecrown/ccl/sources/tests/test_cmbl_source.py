import numpy as np
import pytest

import sacc
import pyccl as ccl

from ..sources import CMBLSource


@pytest.fixture
def cmbl_data():
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

    ell = np.arange(3000)

    sacc_data.add_tracer('Map', 'trc1', quantity='cmb_convergence', spin=0,
                         ell=ell, beam=None)

    return {
        'cosmo': cosmo,
        'sacc_tracer': 'trc1',
        'sacc_data': sacc_data,
        'params': params,
        'systematics': [],
        'systematics_dict': {},
        'scale_': 1.05}


def test_cmbl_source_nosys(lss_data):
    src = CMBLSource(
        sacc_tracer=lss_data['sacc_tracer'],
        scale=0.5)
    src.read(lss_data['sacc_data'])

    src.render(
        lss_data['cosmo'],
        lss_data['params'],
        lss_data['systematics_dict'])

    assert np.allclose(src.scale, 0.5)
    assert np.allclose(src.scale_, 0.5)
    assert src.systematics == []

    assert isinstance(src.tracer_, ccl.CMBLensingTracer)
