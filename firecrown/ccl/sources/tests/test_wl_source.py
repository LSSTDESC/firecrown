import numpy as np
import pytest
import sacc
import pyccl as ccl

from ..sources import WLSource
from ...systematics import MultiplicativeShearBias


@pytest.fixture
def wl_data():
    sacc_data = sacc.Sacc()

    params = dict(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67,
    )
    cosmo = ccl.Cosmology(**params)
    params["blah"] = 0.05
    params["blah2"] = 0.02
    params["blah6"] = 0.51
    wlm = MultiplicativeShearBias(m="blah")

    mn = 0.25
    z = np.linspace(0, 2, 50)
    dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)
    sacc_data.add_tracer("NZ", "trc1", z, dndz)

    # add extra data to make sure nothing weird is pulled back out
    mn = 0.5
    _z = np.linspace(0, 2, 50)
    _dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)
    sacc_data.add_tracer("NZ", "trc2", _z, _dndz)

    return {
        "cosmo": cosmo,
        "sacc_tracer": "trc1",
        "sacc_data": sacc_data,
        "params": params,
        "systematics": ["wlm"],
        "systematics_dict": {"wlm": wlm},
        "scale_": 1.05,
        "z": z,
        "dndz": dndz,
    }


def test_wl_source_nosys(wl_data):
    src = WLSource(sacc_tracer=wl_data["sacc_tracer"], scale=0.5)
    src.read(wl_data["sacc_data"])

    src.render(wl_data["cosmo"], wl_data["params"], wl_data["systematics_dict"])

    assert np.allclose(src.z_, wl_data["z"])
    assert np.allclose(src.dndz_, wl_data["dndz"])
    assert np.allclose(src.scale, 0.5)
    assert np.allclose(src.scale_, 0.5)
    assert src.systematics == []

    assert isinstance(src.tracer_, ccl.WeakLensingTracer)


def test_wl_source_sys(wl_data):
    src = WLSource(
        sacc_tracer=wl_data["sacc_tracer"], systematics=wl_data["systematics"]
    )
    src.read(wl_data["sacc_data"])

    src.render(wl_data["cosmo"], wl_data["params"], wl_data["systematics_dict"])

    assert np.allclose(src.z_, wl_data["z"])
    assert np.allclose(src.dndz_, wl_data["dndz"])
    assert np.allclose(src.scale_, wl_data["scale_"])
    assert src.systematics == wl_data["systematics"]

    assert isinstance(src.tracer_, ccl.WeakLensingTracer)


def test_wl_source_with_ia(wl_data):
    src = WLSource(
        sacc_tracer=wl_data["sacc_tracer"],
        systematics=wl_data["systematics"],
        ia_bias="blah6",
    )
    src.read(wl_data["sacc_data"])

    src.render(wl_data["cosmo"], wl_data["params"], wl_data["systematics_dict"])

    assert np.allclose(src.z_, wl_data["z"])
    assert np.allclose(src.dndz_, wl_data["dndz"])
    assert np.allclose(src.scale_, wl_data["scale_"])
    assert np.allclose(src.ia_bias_, wl_data["params"]["blah6"])
    assert np.shape(src.ia_bias_) == np.shape(src.z_)
    assert src.systematics == wl_data["systematics"]

    assert isinstance(src.tracer_, ccl.WeakLensingTracer)
