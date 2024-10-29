"""Unit tests for the cobaya Mapping connector."""

import pytest
import numpy as np
from cobaya.model import get_model, Model
from cobaya.log import LoggedError
from firecrown.connector.cobaya.ccl import CCLConnector
from firecrown.connector.cobaya.likelihood import LikelihoodConnector
from firecrown.likelihood.likelihood import NamedParameters


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
    # Fiducial parameters for CAMB
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


def test_cobaya_ccl_model(fiducial_params):
    # Fiducial parameters for CAMB
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "test_lk": {
                "external": lambda _self=None: 0.0,
                "requires": {"pyccl_args": None, "pyccl_params": None},
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

    cosmo_args = model_fiducial.provider.get_pyccl_args()
    cosmo_params = model_fiducial.provider.get_pyccl_params()
    assert isinstance(cosmo_args, dict)

    h = fiducial_params["H0"] / 100.0
    assert cosmo_params["h"] * 100.0 == pytest.approx(fiducial_params["H0"], rel=1.0e-5)
    assert cosmo_params["Omega_c"] == pytest.approx(
        fiducial_params["omch2"] / h**2, rel=1.0e-5
    )
    assert cosmo_params["Omega_b"] == pytest.approx(
        fiducial_params["ombh2"] / h**2, rel=1.0e-5
    )
    assert cosmo_params["Omega_k"] == pytest.approx(0.0, rel=1.0e-5)
    assert cosmo_params["A_s"] == pytest.approx(fiducial_params["As"], rel=1.0e-5)
    assert cosmo_params["n_s"] == pytest.approx(fiducial_params["ns"], rel=1.0e-5)
    # The following test fails because of we are using the default
    # neutrino hierarchy, which is normal, while CAMB depends on the
    # parameter which we do not have access to.
    # assert cosmo["m_nu"] == pytest.approx(0.06, rel=1.0e-5)


def test_cobaya_ccl_likelihood(fiducial_params):
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


def test_parameterized_likelihood_missing(fiducial_params):
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "tests/likelihood/lkdir/lk_needing_param.py",
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    with pytest.raises(KeyError):
        _ = get_model(info_fiducial)


def test_parameterized_likelihood_wrong_type(fiducial_params):
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "tests/likelihood/lkdir/lk_needing_param.py",
                "build_parameters": 1.0,
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    with pytest.raises(
        TypeError, match="build_parameters must be a NamedParameters or dict"
    ):
        _ = get_model(info_fiducial)


def test_parameterized_likelihood_dict(fiducial_params):
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "tests/likelihood/lkdir/lk_needing_param.py",
                "build_parameters": {"sacc_filename": "this_sacc_does_not_exist.fits"},
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    model_fiducial = get_model(info_fiducial)
    assert isinstance(model_fiducial, Model)
    assert model_fiducial.logposterior({}).logpost == -1.5


def test_parameterized_likelihood_namedparameters(fiducial_params):
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "tests/likelihood/lkdir/lk_needing_param.py",
                "build_parameters": NamedParameters(
                    {"sacc_filename": "this_sacc_does_not_exist.fits"}
                ),
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    model_fiducial = get_model(info_fiducial)
    assert isinstance(model_fiducial, Model)
    assert model_fiducial.logposterior({}).logpost == -1.5


def test_sampler_parameter_likelihood_missing(fiducial_params):
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "tests/likelihood/lkdir/lk_sampler_parameter.py",
                "build_parameters": NamedParameters({"parameter_prefix": "my_prefix"}),
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    with pytest.raises(LoggedError, match="my_prefix_sampler_param0"):
        _ = get_model(info_fiducial)


def test_sampler_parameter_likelihood(fiducial_params):
    fiducial_params.update({"my_prefix_sampler_param0": 1.0})
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "tests/likelihood/lkdir/lk_sampler_parameter.py",
                "build_parameters": NamedParameters({"parameter_prefix": "my_prefix"}),
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    model_fiducial = get_model(info_fiducial)
    assert isinstance(model_fiducial, Model)
    assert model_fiducial.logposterior({}).logpost == -2.1


def test_derived_parameter_likelihood(fiducial_params):
    fiducial_params.update({"derived_section__derived_param0": {"derived": True}})
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "tests/likelihood/lkdir/lk_derived_parameter.py",
                "derived_parameters": ["derived_section__derived_param0"],
            }
        },
        "theory": {
            "camb": {"extra_args": {"num_massive_neutrinos": 1}},
            "fcc_ccl": {"external": CCLConnector, "input_style": "CAMB"},
        },
    }

    model_fiducial = get_model(info_fiducial)
    assert isinstance(model_fiducial, Model)
    logpost = model_fiducial.logposterior({})
    assert logpost.logpost == -3.14
    assert logpost.derived[0] == 1.0


def test_default_factory():
    fiducial_params = {
        "ia_bias": 1.0,
        "sigma8": 0.8,
        "alphaz": 1.0,
        "h": 0.7,
        "lens0_bias": 1.0,
        "lens0_delta_z": 0.1,
        "lens1_bias": 1.0,
        "lens1_delta_z": 0.1,
        "lens2_bias": 1.0,
        "lens2_delta_z": 0.1,
        "lens3_bias": 1.0,
        "lens3_delta_z": 0.1,
        "lens4_delta_z": 0.1,
        "lens4_bias": 1.0,
        "m_nu": 0.06,
        "n_s": 0.96,
        "Neff": 3.046,
        "Omega_b": 0.048,
        "Omega_c": 0.26,
        "Omega_k": 0.0,
        "src0_delta_z": 0.1,
        "src0_mult_bias": 1.0,
        "src1_delta_z": 0.1,
        "src1_mult_bias": 1.0,
        "src2_delta_z": 0.1,
        "src2_mult_bias": 1.0,
        "src3_delta_z": 0.1,
        "src3_mult_bias": 1.0,
        "w0": -1.0,
        "wa": 0.0,
        "z_piv": 0.3,
        "T_CMB": 2.7255,
    }
    info_fiducial = {
        "params": fiducial_params,
        "likelihood": {
            "lk_connector": {
                "external": LikelihoodConnector,
                "firecrownIni": "firecrown.likelihood.factories.build_two_point_likelihood",
                "build_parameters": {
                    "likelihood_config": "examples/des_y1_3x2pt/pure_ccl_experiment.yaml"
                },
            },
        },
    }
    model_fiducial = get_model(info_fiducial)
    assert isinstance(model_fiducial, Model)
    logpost = model_fiducial.logposterior({})
    assert np.isfinite(logpost.logpost)
