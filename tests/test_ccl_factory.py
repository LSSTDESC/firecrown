"""Test the CCLFactory object."""

import re
import numpy as np
import pytest
import pyccl
import pyccl.modified_gravity
from pyccl.neutrinos import NeutrinoMassSplits
import pydantic

from firecrown.ccl_factory import (
    CAMBExtraParams,
    CCLCalculatorArgs,
    CCLCreationMode,
    CCLFactory,
    MuSigmaModel,
    PoweSpecAmplitudeParameter,
    CCLSplineParams,
)
from firecrown.updatable import get_default_params_map
from firecrown.parameters import ParamsMap
from firecrown.utils import base_model_from_yaml, base_model_to_yaml


@pytest.fixture(name="amplitude_parameter", params=list(PoweSpecAmplitudeParameter))
def fixture_amplitude_parameter(request):
    return request.param


@pytest.fixture(name="neutrino_mass_splits", params=list(NeutrinoMassSplits))
def fixture_neutrino_mass_splits(request):
    return request.param


@pytest.fixture(
    name="require_nonlinear_pk",
    params=[True, False],
    ids=["require_nonlinear_pk", "no_require_nonlinear_pk"],
)
def fixture_require_nonlinear_pk(request):
    return request.param


@pytest.fixture(
    name="ccl_creation_mode",
    params=[
        CCLCreationMode.DEFAULT,
        CCLCreationMode.PURE_CCL_MODE,
        CCLCreationMode.MU_SIGMA_ISITGR,
    ],
    ids=["default", "pure_ccl_mode", "mu_sigma_isitgr"],
)
def fixture_ccl_creation_mode(request):
    return request.param


@pytest.fixture(
    name="camb_extra_params",
    params=[
        None,
        {"halofit_version": "mead"},
        {"halofit_version": "mead", "kmax": 0.1},
        {"halofit_version": "mead", "kmax": 0.1, "lmax": 100},
        {"dark_energy_model": "ppf"},
    ],
    ids=["default", "mead", "mead_kmax", "mead_kmax_lmax", "ppf"],
)
def fixture_camb_extra_params(request) -> CAMBExtraParams | None:
    return (
        CAMBExtraParams.model_validate(request.param)
        if request.param is not None
        else None
    )


@pytest.fixture(
    name="ccl_spline_params",
    params=[None, {"a_spline_na": 451}],
    ids=["default", "set_a_spline_na"],
)
def fixture_ccl_spline_params(request) -> CCLSplineParams | None:
    return CCLSplineParams(**request.param) if request.param is not None else None


Z_ARRAY = np.linspace(0.0, 5.0, 100, dtype=np.float64)
A_ARRAY = np.array(1.0 / (1.0 + np.flip(Z_ARRAY)), dtype=np.float64)
K_ARRAY = np.geomspace(1.0e-5, 10.0, 100, dtype=np.float64)
A_GRID, K_GRID = np.meshgrid(A_ARRAY, K_ARRAY, indexing="ij")

CHI_ARRAY = np.linspace(100.0, 0.0, 100, dtype=np.float64)
H_OVER_H0_ARRAY = np.linspace(1.0, 100.0, 100, dtype=np.float64)
# Simple power spectrum model for testing
PK_ARRAY = (
    2.0e-9
    * (K_GRID / 0.05) ** (0.96)
    * (np.log(1 + 2.34 * K_GRID / 0.30) / (2.34 * K_GRID / 0.30))
    * (1 / (1 + (K_GRID / (0.1 * 0.05)) ** (3 * 0.96)) * A_GRID)
)

BACKGROUND = {
    "background": {"a": A_ARRAY, "chi": CHI_ARRAY, "h_over_h0": H_OVER_H0_ARRAY}
}

PK_LINEAR = {
    "pk_linear": {"k": K_ARRAY, "a": A_ARRAY, "delta_matter:delta_matter": PK_ARRAY}
}

PK_NONLIN = {
    "pk_nonlin": {
        "k": np.linspace(0.1, 1.0, 100),
        "a": np.linspace(0.1, 1.0, 100),
        "delta_matter:delta_matter": PK_ARRAY,
    }
}


@pytest.fixture(
    name="calculator_args",
    params=[BACKGROUND, BACKGROUND | PK_LINEAR, BACKGROUND | PK_LINEAR | PK_NONLIN],
    ids=["background", "background_linear_pk", "background_linear_pk_nonlinear_pk"],
)
def fixture_calculator_args(request) -> CCLCalculatorArgs:
    return request.param


def test_setting_each_spline_param(
    ccl_creation_mode: CCLCreationMode,
    camb_extra_params: CAMBExtraParams | None,
) -> None:
    # test_helper is a closure that captures the values of ccl_creation mode
    # and camb_extra_params, and is callable with just the param_name and
    # param_value to be tested.
    def test_helper(param_name: str, param_value: float | int):
        original_param_value = getattr(pyccl.spline_params, param_name.upper())
        args = {param_name: param_value}
        spline_params = CCLSplineParams(**args)  # type: ignore
        assert original_param_value != param_value
        ccl_factory = CCLFactory(
            amplitude_parameter=PoweSpecAmplitudeParameter.AS,
            mass_split=NeutrinoMassSplits.NORMAL,
            require_nonlinear_pk=False,
            creation_mode=ccl_creation_mode,
            camb_extra_params=camb_extra_params,
            ccl_spline_params=spline_params,
        )
        default_params = get_default_params_map(ccl_factory)
        ccl_factory.update(default_params)
        cosmo = ccl_factory.create()
        assert cosmo is not None
        assert getattr(cosmo.cosmo.spline_params, param_name.upper()) == param_value
        assert getattr(pyccl.spline_params, param_name.upper()) == original_param_value

    test_helper("a_spline_na", 73)
    test_helper("a_spline_min", 0.003)
    test_helper("a_spline_minlog_pk", 0.2)
    test_helper("a_spline_min_pk", 0.5)
    test_helper("a_spline_minlog_sm", 0.02)
    test_helper("a_spline_min_sm", 0.05)
    test_helper("a_spline_minlog", 0.02)
    test_helper("a_spline_nlog", 2112)

    test_helper("logm_spline_delta", 0.02)
    test_helper("logm_spline_nm", 42)
    test_helper("logm_spline_min", 4)
    test_helper("logm_spline_max", 21)

    test_helper("a_spline_na_sm", 11)
    test_helper("a_spline_nlog_sm", 8)
    test_helper("a_spline_na_pk", 44)
    test_helper("a_spline_nlog_sm", 9)

    test_helper("k_max_spline", 52)
    test_helper("k_max", 999)
    test_helper("k_min", 0.01)
    test_helper("dlogk_integration", 0.2)
    test_helper("dchi_integration", 5.5)
    test_helper("n_k", 180)
    test_helper("n_k_3dcor", 9999)

    test_helper("ell_min_corr", 0.1)
    test_helper("ell_max_corr", 5000)
    test_helper("n_ell_corr", 4444)


def test_ccl_spline_params_validation():
    with pytest.raises(pydantic.ValidationError):
        _ = CCLSplineParams(logm_spline_min=20, logm_spline_max=10)
    with pytest.raises(pydantic.ValidationError):
        _ = CCLSplineParams(k_min=0.5, k_max=0.1)
    with pytest.raises(pydantic.ValidationError):
        _ = CCLSplineParams(ell_min_corr=100, ell_max_corr=10)


def test_ccl_spline_params_context_manager_exception():
    with CCLSplineParams(k_max=900) as params:
        assert params.k_max == 900
    with pytest.raises(AssertionError):
        with CCLSplineParams(n_ell_corr=1000):
            assert False


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def test_ccl_factory_simple(
    amplitude_parameter: PoweSpecAmplitudeParameter,
    neutrino_mass_splits: NeutrinoMassSplits,
    require_nonlinear_pk: bool,
    ccl_creation_mode: CCLCreationMode,
    camb_extra_params: CAMBExtraParams | None,
    ccl_spline_params: CCLSplineParams | None,
) -> None:
    ccl_factory = CCLFactory(
        amplitude_parameter=amplitude_parameter,
        mass_split=neutrino_mass_splits,
        require_nonlinear_pk=require_nonlinear_pk,
        creation_mode=ccl_creation_mode,
        camb_extra_params=camb_extra_params,
        ccl_spline_params=ccl_spline_params,
    )

    assert ccl_factory is not None
    assert ccl_factory.amplitude_parameter == amplitude_parameter
    assert ccl_factory.mass_split == neutrino_mass_splits
    assert ccl_factory.require_nonlinear_pk == require_nonlinear_pk

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)
    if ccl_spline_params is not None:
        for key, value in ccl_spline_params.model_dump().items():
            if value is not None:
                assert (
                    # pylint: disable-next=protected-access
                    cosmo._spline_params[key.upper()]
                    == value
                )


def test_ccl_factory_ccl_args(
    amplitude_parameter: PoweSpecAmplitudeParameter,
    neutrino_mass_splits: NeutrinoMassSplits,
    require_nonlinear_pk: bool,
    calculator_args: CCLCalculatorArgs,
    ccl_spline_params: CCLSplineParams,
) -> None:
    ccl_factory = CCLFactory(
        amplitude_parameter=amplitude_parameter,
        mass_split=neutrino_mass_splits,
        require_nonlinear_pk=require_nonlinear_pk,
        ccl_spline_params=ccl_spline_params,
    )

    if require_nonlinear_pk and "pk_linear" not in calculator_args:
        pytest.skip(
            "Nonlinear power spectrum requested but "
            "linear power spectrum not provided."
        )

    assert ccl_factory is not None
    assert ccl_factory.amplitude_parameter == amplitude_parameter
    assert ccl_factory.mass_split == neutrino_mass_splits
    assert ccl_factory.require_nonlinear_pk == require_nonlinear_pk

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create(calculator_args)

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)
    if ccl_spline_params is not None:
        for key, value in ccl_spline_params.model_dump().items():
            if value is not None:
                # pylint: disable-next=protected-access
                assert cosmo._spline_params[key.upper()] == value


def test_ccl_factory_update() -> None:
    ccl_factory = CCLFactory()

    assert ccl_factory is not None

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)

    new_params = ParamsMap(default_params.copy())
    new_params["Omega_c"] = 0.1

    ccl_factory.reset()
    ccl_factory.update(new_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)


def test_ccl_factory_amplitude_parameter(
    amplitude_parameter: PoweSpecAmplitudeParameter,
) -> None:
    ccl_factory = CCLFactory(amplitude_parameter=amplitude_parameter)

    assert ccl_factory is not None

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)


def test_ccl_factory_neutrino_mass_splits(
    neutrino_mass_splits: NeutrinoMassSplits,
) -> None:
    ccl_factory = CCLFactory(mass_split=neutrino_mass_splits)

    assert ccl_factory is not None
    assert ccl_factory.mass_split == neutrino_mass_splits

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)


def test_ccl_factory_ccl_creation_mode(
    ccl_creation_mode: CCLCreationMode,
) -> None:
    ccl_factory = CCLFactory(creation_mode=ccl_creation_mode)

    assert ccl_factory is not None
    assert ccl_factory.creation_mode == CCLCreationMode(ccl_creation_mode)

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)


def test_ccl_factory_get() -> None:
    ccl_factory = CCLFactory()

    assert ccl_factory is not None

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)

    cosmo2 = ccl_factory.get()

    assert cosmo2 is not None
    assert isinstance(cosmo2, pyccl.Cosmology)

    assert cosmo is cosmo2


def test_ccl_factory_get_not_created() -> None:
    ccl_factory = CCLFactory()

    with pytest.raises(ValueError, match="CCLFactory object has not been created yet."):
        ccl_factory.get()


def test_ccl_factory_create_twice() -> None:
    ccl_factory = CCLFactory()

    assert ccl_factory is not None

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)

    with pytest.raises(ValueError, match="CCLFactory object has already been created."):
        ccl_factory.create()


def test_ccl_factory_create_not_updated() -> None:
    ccl_factory = CCLFactory()

    assert ccl_factory is not None

    with pytest.raises(ValueError, match="Parameters have not been updated yet."):
        ccl_factory.create()


def test_ccl_factory_tofrom_yaml() -> None:
    ccl_factory = CCLFactory()

    assert ccl_factory is not None

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)

    yaml_str = base_model_to_yaml(ccl_factory)

    ccl_factory2 = base_model_from_yaml(CCLFactory, yaml_str)

    assert ccl_factory2 is not None
    ccl_factory2.update(default_params)

    cosmo2 = ccl_factory2.create()

    assert cosmo2 is not None
    assert isinstance(cosmo2, pyccl.Cosmology)

    assert cosmo == cosmo2


def test_ccl_factory_tofrom_yaml_all_options(
    amplitude_parameter: PoweSpecAmplitudeParameter,
    neutrino_mass_splits: NeutrinoMassSplits,
    require_nonlinear_pk: bool,
) -> None:
    ccl_factory = CCLFactory(
        amplitude_parameter=amplitude_parameter,
        mass_split=neutrino_mass_splits,
        require_nonlinear_pk=require_nonlinear_pk,
    )

    assert ccl_factory is not None
    assert ccl_factory.amplitude_parameter == amplitude_parameter
    assert ccl_factory.mass_split == neutrino_mass_splits
    assert ccl_factory.require_nonlinear_pk == require_nonlinear_pk

    default_params = get_default_params_map(ccl_factory)

    ccl_factory.update(default_params)

    cosmo = ccl_factory.create()

    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)

    yaml_str = base_model_to_yaml(ccl_factory)

    ccl_factory2 = base_model_from_yaml(CCLFactory, yaml_str)

    assert ccl_factory2 is not None
    assert ccl_factory2.amplitude_parameter == amplitude_parameter
    assert ccl_factory2.mass_split == neutrino_mass_splits
    assert ccl_factory2.require_nonlinear_pk == require_nonlinear_pk
    ccl_factory2.update(default_params)

    cosmo2 = ccl_factory2.create()

    assert cosmo2 is not None
    assert isinstance(cosmo2, pyccl.Cosmology)

    assert cosmo == cosmo2


def test_ccl_factory_invalid_amplitude_parameter() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Value error, 'Im not a valid value' is not a valid "
            "PoweSpecAmplitudeParameter"
        ),
    ):
        CCLFactory(amplitude_parameter="Im not a valid value")


def test_ccl_factory_invalid_mass_splits() -> None:
    with pytest.raises(
        ValueError,
        match=".*Invalid value for NeutrinoMassSplits: Im not a valid value.*",
    ):
        CCLFactory(mass_split="Im not a valid value")


def test_ccl_factory_invalid_creation_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Value error, 'Im not a valid value' is not a valid CCLCreationMode",
    ):
        CCLFactory(creation_mode="Im not a valid value")


def test_ccl_factory_from_dict() -> None:
    ccl_factory_dict = {
        "amplitude_parameter": PoweSpecAmplitudeParameter.SIGMA8,
        "mass_split": NeutrinoMassSplits.EQUAL,
        "require_nonlinear_pk": True,
        "creation_mode": CCLCreationMode.DEFAULT,
    }

    ccl_factory = CCLFactory.model_validate(ccl_factory_dict)

    assert ccl_factory is not None
    assert ccl_factory.amplitude_parameter == PoweSpecAmplitudeParameter.SIGMA8
    assert ccl_factory.mass_split == NeutrinoMassSplits.EQUAL
    assert ccl_factory.require_nonlinear_pk is True


def test_ccl_factory_from_dict_wrong_type() -> None:
    ccl_factory_dict = {
        "amplitude_parameter": 0.32,
        "mass_split": NeutrinoMassSplits.EQUAL,
        "require_nonlinear_pk": True,
    }

    with pytest.raises(
        ValueError,
        match=".*Input should be.*",
    ):
        CCLFactory.model_validate(ccl_factory_dict)


def test_ccl_factory_camb_extra_params_invalid() -> None:
    with pytest.raises(
        ValueError,
        match=".*validation error for CCLFactory*",
    ):
        CCLFactory(camb_extra_params="Im not a valid value")


def test_ccl_factory_camb_extra_params_invalid_model() -> None:
    ccl_factory = CCLFactory(
        camb_extra_params={"dark_energy_model": "Im not a valid value"}
    )
    params = get_default_params_map(ccl_factory)
    ccl_factory.update(params)
    ccl_cosmo = ccl_factory.create()
    with pytest.raises(
        ValueError,
        match="The only dark energy models CCL supports with CAMB are fluid and ppf.",
    ):
        ccl_cosmo.compute_linear_power()


def test_ccl_factory_invalid_extra_params() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid parameters: {'not_a_valid_param'}"),
    ):
        CCLFactory(not_a_valid_param="Im not a valid value")


def test_validate_creation_mode_incompatible():
    ccl_factory = CCLFactory(creation_mode=CCLCreationMode.PURE_CCL_MODE)
    params = get_default_params_map(ccl_factory)
    ccl_factory.update(params)
    with pytest.raises(
        ValueError,
        match="Calculator Mode can only be used with the DEFAULT "
        "creation mode and no CAMB extra parameters.",
    ):
        ccl_factory.create(
            calculator_args=CCLCalculatorArgs(
                background={
                    "a": A_ARRAY,
                    "chi": CHI_ARRAY,
                    "h_over_h0": H_OVER_H0_ARRAY,
                }
            )
        )


def test_mu_sigma_model() -> None:
    mu_sigma_model = MuSigmaModel()

    assert mu_sigma_model is not None

    default_params = get_default_params_map(mu_sigma_model)

    mu_sigma_model.update(default_params)

    musigma = mu_sigma_model.create()

    assert musigma is not None
    assert isinstance(musigma, pyccl.modified_gravity.MuSigmaMG)


def test_mu_sigma_create_not_updated() -> None:
    mu_sigma_model = MuSigmaModel()

    assert mu_sigma_model is not None

    with pytest.raises(ValueError, match="Parameters have not been updated yet."):
        mu_sigma_model.create()
