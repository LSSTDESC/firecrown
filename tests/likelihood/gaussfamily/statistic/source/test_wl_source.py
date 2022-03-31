import numpy as np
import pytest
import sacc
import pyccl as ccl

from ..sources import WLSource
from ...systematics import MultiplicativeShearBias
from ..wl_source import get_from_prefix_param, WLSourceSystematic


@pytest.fixture()
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


@pytest.fixture()
def empty_params():
    return {}


@pytest.fixture()
def params_with_generic_name():
    return {"bias": 1.5}


@pytest.fixture()
def params_with_specific_name():
    return {"thing3_bias": 2.5}


@pytest.fixture()
def params_with_both_names(params_with_generic_name, params_with_specific_name):
    return params_with_specific_name | params_with_generic_name


def test_get_from_prefix_param_exception_for_empty_params(empty_params):
    systematic = WLSourceSystematic()
    with pytest.raises(KeyError) as info:
        get_from_prefix_param(systematic, empty_params,
                              "thing3", "bias")
    assert info.type is KeyError
    # The expected string formatting is ugly, because of the expected
    # double-quote and single-quote characters that will be in the error
    # message.
    assert info.exconly() == "KeyError: \"WLSourceSystematic key `bias' not " \
                             "found\""


def test_get_prefix_param_finds_specific_name(params_with_specific_name):
    systematic = WLSourceSystematic
    assert get_from_prefix_param(systematic, params_with_specific_name,
                                 "thing3", "bias") == 2.5


def test_get_prefix_param_finds_generic_name(params_with_generic_name):
    systematic = WLSourceSystematic
    assert get_from_prefix_param(systematic, params_with_generic_name,
                                 "thing3", "bias") == 1.5


def test_get_prefix_param_prefers_specific_name(params_with_both_names):
    systematic = WLSourceSystematic
    assert get_from_prefix_param(systematic, params_with_both_names,
                                 "thing3", "bias") == 2.5


def test_get_prefix_param_skips_wrong_prefix(params_with_both_names):
    systematic = WLSourceSystematic
    assert get_from_prefix_param(systematic, params_with_both_names,
                                 "thing0", "bias") == 1.5


def test_get_prefix_params_with_no_prefix_finds_general(params_with_both_names):
    systematic = WLSourceSystematic
    assert get_from_prefix_param(systematic, params_with_both_names,
                                 None, "bias") == 1.5


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
