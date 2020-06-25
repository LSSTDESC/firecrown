import numpy as np

import sacc
import pyccl as ccl
import pytest

from ..sources import ClusterSource
from ...systematics import PowerLawMOR


@pytest.fixture
def cl_data():
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

    mn = 0.25
    z = np.linspace(0, 2, 50)
    dndz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)
    sacc_data.add_tracer(
        'NZ', 'trc1', z, dndz,
        metadata={"lnlam_min": -12, "lnlam_max": -10, 'area_sd': 200})

    # add extra data to make sure nothing weird is pulled back out
    mn = 0.5
    _z = np.linspace(0, 2, 50)
    _dndz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)
    sacc_data.add_tracer('NZ', 'trc2', _z, _dndz)

    return {
        'cosmo': cosmo,
        'sacc_tracer': 'trc1',
        'sacc_data': sacc_data,
        'params': params,
        'z': z,
        'dndz': dndz}


def test_cl_source(cl_data):
    src = ClusterSource(sacc_tracer=cl_data['sacc_tracer'])
    src.read(cl_data['sacc_data'])

    assert np.allclose(src.lnlam_min_orig, -12)
    assert np.allclose(src.lnlam_max_orig, -10)
    assert np.allclose(src.area_sr_orig, 200 * (np.pi/180.0)**2)

    src.render(
        cl_data['cosmo'],
        cl_data['params'],
        {})

    assert np.allclose(src.z_, cl_data['z'])
    assert np.allclose(src.dndz_, cl_data['dndz'])
    assert np.allclose(src.scale, 1.0)
    assert np.allclose(src.scale_, 1.0)
    assert src.systematics == []

    assert np.allclose(src.lnlam_min_, -12)
    assert np.allclose(src.lnlam_max_, -10)
    assert np.allclose(src.area_sr_, 200 * (np.pi/180.0)**2)

    assert hasattr(src, "mor_")
    assert hasattr(src, "inv_mor_")
    assert hasattr(src, "selfunc_")


def test_cl_source_override_sys(cl_data):
    src = ClusterSource(sacc_tracer=cl_data['sacc_tracer'], systematics=["pl"])
    src.read(cl_data['sacc_data'])

    assert np.allclose(src.lnlam_min_orig, -12)
    assert np.allclose(src.lnlam_max_orig, -10)
    assert np.allclose(src.area_sr_orig, 200 * (np.pi/180.0)**2)

    sys = PowerLawMOR(lnlam_norm="a", mass_slope="b", a_slope="c")
    cl_data['params']["a"] = 1.0
    cl_data['params']["b"] = 2.0
    cl_data['params']["c"] = 4.0
    src.render(
        cl_data['cosmo'],
        cl_data['params'],
        {"pl": sys})

    assert np.allclose(src.z_, cl_data['z'])
    assert np.allclose(src.dndz_, cl_data['dndz'])
    assert np.allclose(src.scale, 1.0)
    assert np.allclose(src.scale_, 1.0)
    assert src.systematics == ["pl"]

    assert np.allclose(src.lnlam_min_, -12)
    assert np.allclose(src.lnlam_max_, -10)
    assert np.allclose(src.area_sr_, 200 * (np.pi/180.0)**2)

    assert hasattr(src, "mor_")
    assert hasattr(src, "inv_mor_")
    assert hasattr(src, "selfunc_")

    assert np.allclose(
        src.mor_(10, 0.3),
        1 + 2 * (10 - np.log(1e14)) + 4 * np.log(0.3),
    )

    lnm_min = src.inv_mor_(src.lnlam_min_, 0.5)
    lnm_max = src.inv_mor_(src.lnlam_max_, 0.5)

    assert np.array_equal(
        src.selfunc_((lnm_min + lnm_max)/2, 0.5),
        np.atleast_2d(src.dndz_interp_(1/0.5-1)),
    )

    assert np.array_equal(src.selfunc_(lnm_min - 10, 0.5), np.atleast_2d(0))
    assert np.array_equal(src.selfunc_(lnm_max + 10, 0.5), np.atleast_2d(0))
