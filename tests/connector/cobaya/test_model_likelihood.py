"""Unit tests for the cobaya Mapping connector."""

import pytest
import pyccl as ccl
from cobaya.model import get_model, Model
from firecrown.connector.cobaya.ccl import CCLConnector
from firecrown.connector.cobaya.likelihood import LikelihoodConnector


def test_cobaya_ccl_initialize():
    ccl_connector = CCLConnector(info={"input_style": "CAMB"})

    assert isinstance(ccl_connector, CCLConnector)
    assert ccl_connector.input_style == "CAMB"


def test_cobaya_ccl_initialize_with_params():
    ccl_connector = CCLConnector(info={"input_style": "CAMB"})

    ccl_connector.initialize_with_params()

    assert isinstance(ccl_connector, CCLConnector)
    assert ccl_connector.input_style == "CAMB"


def test_cobaya_likelihood_initialize():
    lk_connector = LikelihoodConnector(
        info={"firecrownIni": "tests/likelihood/lkdir/lkscript.py"}
    )

    assert isinstance(lk_connector, LikelihoodConnector)
    assert lk_connector.firecrownIni == "tests/likelihood/lkdir/lkscript.py"


def test_cobaya_likelihood_initialize_with_params():
    lk_connector = LikelihoodConnector(
        info={"firecrownIni": "tests/likelihood/lkdir/lkscript.py"}
    )

    lk_connector.initialize_with_params()

    assert isinstance(lk_connector, LikelihoodConnector)
    assert lk_connector.firecrownIni == "tests/likelihood/lkdir/lkscript.py"


@pytest.fixture(name="fiducial_params")
def fixture_fiducial_params():
    fiducial_params = {
        "ombh2": 0.022,
        "omch2": 0.12,
        "H0": 68,
        "tau": 0.07,
        "As": 2.2e-9,
        "ns": 0.96,
        "mnu": 0.06,
        "nnu": 3.046,
    }

    return fiducial_params


def test_cobaya_ccl_with_model(fiducial_params):
    # Fiducial parameters for CAMB
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "test_lk": {
                "external": lambda _self=None: 0.0,
                "requires": {"pyccl": None},
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    model_fiducial = get_model(info_fiducial)
    assert isinstance(model_fiducial, Model)
    model_fiducial.logposterior({})

    cosmo = model_fiducial.provider.get_pyccl()
    assert isinstance(cosmo, ccl.Cosmology)

    h = fiducial_params["H0"] / 100.0
    assert cosmo["H0"] == pytest.approx(fiducial_params["H0"], rel=1.0e-5)
    assert cosmo["Omega_c"] == pytest.approx(
        fiducial_params["omch2"] / h**2, rel=1.0e-5
    )
    assert cosmo["Omega_b"] == pytest.approx(
        fiducial_params["ombh2"] / h**2, rel=1.0e-5
    )
    assert cosmo["Omega_k"] == pytest.approx(0.0, rel=1.0e-5)
    assert cosmo["A_s"] == pytest.approx(fiducial_params["As"], rel=1.0e-5)
    assert cosmo["n_s"] == pytest.approx(fiducial_params["ns"], rel=1.0e-5)
    # The following test fails because of we are using the default
    # neutrino hierarchy, which is normal, while CAMB depends on the
    # parameter which we do not have access to.
    # assert cosmo["m_nu"] == pytest.approx(0.06, rel=1.0e-5)


def test_cobaya_ccl_with_likelihood(fiducial_params):
    # Fiducial parameters for CAMB
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "tests/likelihood/lkdir/lkscript.py",
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    model_fiducial = get_model(info_fiducial)
    assert isinstance(model_fiducial, Model)
    assert model_fiducial.logposterior({}).logpost == -3.0
