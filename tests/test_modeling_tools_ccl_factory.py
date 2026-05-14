"""Test the CCLFactory object."""

import re
import itertools as it
import numpy as np
from numpy.testing import assert_allclose
import pytest
import pyccl
import pyccl.modified_gravity
from pyccl.neutrinos import NeutrinoMassSplits
import pydantic

from firecrown.modeling_tools import (
    CAMBExtraParams,
    CCLCalculatorArgs,
    CCLCreationMode,
    CCLPureModeTransferFunction,
    CCLFactory,
    MuSigmaModel,
    PoweSpecAmplitudeParameter,
    CCLSplineParams,
)
from firecrown.updatable import get_default_params_map
from firecrown.updatable import ParamsMap
from firecrown.utils import base_model_from_yaml, base_model_to_yaml
from firecrown.modeling_tools import ModelingTools

# pylint: disable=too-many-lines


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
)
def fixture_ccl_creation_mode(request) -> CCLCreationMode:
    """Fixture providing each CCLCreationMode."""
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
    ids=["no_camb", "mead", "mead_kmax", "mead_kmax_lmax", "ppf"],
)
def fixture_camb_extra_params(request) -> CAMBExtraParams | None:
    """Fixture providing each possible CAMBExtraParams configuration."""
    if request.param is not None:
        return CAMBExtraParams.model_validate(request.param)
    return None


@pytest.fixture(
    name="ccl_creation_mode_and_camb_params",
    params=[
        # Test all creation modes with no CAMB params
        (CCLCreationMode.DEFAULT, None, "default-no_camb"),
        (CCLCreationMode.PURE_CCL_MODE, None, "pure_ccl_mode-no_camb"),
        (CCLCreationMode.MU_SIGMA_ISITGR, None, "mu_sigma_isitgr-no_camb"),
        # Test PURE_CCL_MODE with all CAMB param variations
        (
            CCLCreationMode.PURE_CCL_MODE,
            {"halofit_version": "mead"},
            "pure_ccl_mode-mead",
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            {"halofit_version": "mead", "kmax": 0.1},
            "pure_ccl_mode-mead_kmax",
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            {"halofit_version": "mead", "kmax": 0.1, "lmax": 100},
            "pure_ccl_mode-mead_kmax_lmax",
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            {"dark_energy_model": "ppf"},
            "pure_ccl_mode-ppf",
        ),
    ],
    ids=lambda x: x[2],  # Use the third element (id string) as the test id
)
def fixture_ccl_creation_mode_and_camb_params(request):
    """Provides valid combinations of creation mode and CAMB params.

    CAMB extra parameters are only compatible with PURE_CCL_MODE, so this
    fixture only generates valid combinations, eliminating pytest.skip.

    Returns:
        tuple: (CCLCreationMode, CAMBExtraParams | None)
    """
    creation_mode, camb_params_dict, _ = request.param
    camb_params = (
        CAMBExtraParams.model_validate(camb_params_dict)
        if camb_params_dict is not None
        else None
    )
    return creation_mode, camb_params


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


@pytest.fixture(
    name="require_nonlinear_pk_and_calculator_args",
    params=[
        # When nonlinear PK is NOT required, any calculator_args work
        (False, BACKGROUND, "no_require_nonlinear_pk-background"),
        (False, BACKGROUND | PK_LINEAR, "no_require_nonlinear_pk-background_linear_pk"),
        (
            False,
            BACKGROUND | PK_LINEAR | PK_NONLIN,
            "no_require_nonlinear_pk-background_linear_pk_nonlinear_pk",
        ),
        # When nonlinear PK IS required, only calculator_args with pk_linear work
        (
            True,
            BACKGROUND | PK_LINEAR,
            "require_nonlinear_pk-background_linear_pk",
        ),
        (
            True,
            BACKGROUND | PK_LINEAR | PK_NONLIN,
            "require_nonlinear_pk-background_linear_pk_nonlinear_pk",
        ),
    ],
    ids=lambda x: x[2],  # Use the third element (id string) as the test id
)
def fixture_require_nonlinear_pk_and_calculator_args(request):
    """Fixture providing valid combinations of require_nonlinear_pk and calculator_args.

    Nonlinear PK requires linear PK to be present, so this fixture only generates
    valid combinations, eliminating the need for pytest.skip.

    Returns:
        tuple: (bool, CCLCalculatorArgs) for require_nonlinear_pk and calculator_args
    """
    require_nl_pk, calc_args, _ = request.param
    return require_nl_pk, calc_args


@pytest.mark.parametrize(
    "transfer_function",
    list(CCLPureModeTransferFunction),
)
def test_ccl_factory_pure_mode_transfer_function(transfer_function):
    ccl_factory = CCLFactory(
        creation_mode=CCLCreationMode.PURE_CCL_MODE,
        pure_ccl_transfer_function=transfer_function,
    )
    assert ccl_factory.pure_ccl_transfer_function == transfer_function
    assert ccl_factory.creation_mode == CCLCreationMode.PURE_CCL_MODE

    params = get_default_params_map(ccl_factory)
    ccl_factory.update(params)
    cosmo = ccl_factory.create()
    assert isinstance(cosmo, pyccl.Cosmology)


def test_ccl_factory_invalid_pure_mode_transfer_function():
    with pytest.raises(ValueError, match="is not a valid CCLPureModeTransferFunction"):
        CCLFactory(
            creation_mode=CCLCreationMode.PURE_CCL_MODE,
            pure_ccl_transfer_function="Im not a valid value",
        )


def test_setting_each_spline_param(
    ccl_creation_mode_and_camb_params: tuple[CCLCreationMode, CAMBExtraParams | None],
) -> None:
    # test_helper is a closure that captures the values of ccl_creation mode
    # and camb_extra_params, and is callable with just the param_name and
    # param_value to be tested.
    ccl_creation_mode, camb_extra_params = ccl_creation_mode_and_camb_params

    def test_helper(param_name: str, param_value: float | int):
        # Using combined fixture ensures only valid combinations are tested
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


def _nu_mass_is_list(mass_split: NeutrinoMassSplits):
    """Check if the neutrino mass split is a list.

    This function is used to check if the neutrino mass split is a list or a sum.
    """
    assert mass_split in NeutrinoMassSplits
    return mass_split in (NeutrinoMassSplits.LIST, NeutrinoMassSplits.SUM)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def test_ccl_factory_simple(
    amplitude_parameter: PoweSpecAmplitudeParameter,
    neutrino_mass_splits: NeutrinoMassSplits,
    require_nonlinear_pk: bool,
    ccl_creation_mode_and_camb_params: tuple[CCLCreationMode, CAMBExtraParams | None],
    ccl_spline_params: CCLSplineParams | None,
) -> None:
    """Test CCL factory with various parameter combinations.

    This test uses a combined fixture that only generates valid combinations
    of creation mode and CAMB parameters, eliminating unnecessary skips.
    """
    ccl_creation_mode, camb_extra_params = ccl_creation_mode_and_camb_params

    ccl_factory = CCLFactory(
        amplitude_parameter=amplitude_parameter,
        mass_split=neutrino_mass_splits,
        require_nonlinear_pk=require_nonlinear_pk,
        creation_mode=ccl_creation_mode,
        camb_extra_params=camb_extra_params,
        ccl_spline_params=ccl_spline_params,
        num_neutrino_masses=(3 if _nu_mass_is_list(neutrino_mass_splits) else None),
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
    require_nonlinear_pk_and_calculator_args: tuple[bool, CCLCalculatorArgs],
    ccl_spline_params: CCLSplineParams,
) -> None:
    """Test CCL factory with calculator args.

    Uses combined fixture to only test valid combinations where nonlinear PK
    requirements are compatible with provided calculator args.
    """
    require_nonlinear_pk, calculator_args = require_nonlinear_pk_and_calculator_args

    ccl_factory = CCLFactory(
        amplitude_parameter=amplitude_parameter,
        mass_split=neutrino_mass_splits,
        require_nonlinear_pk=require_nonlinear_pk,
        ccl_spline_params=ccl_spline_params,
        num_neutrino_masses=(3 if _nu_mass_is_list(neutrino_mass_splits) else None),
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

    # TODO: should the following line use a deep copy?
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
    ccl_factory = CCLFactory(
        mass_split=neutrino_mass_splits,
        num_neutrino_masses=(3 if _nu_mass_is_list(neutrino_mass_splits) else None),
    )

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
    num_neutrino_masses = 3 if _nu_mass_is_list(neutrino_mass_splits) else None

    ccl_factory = CCLFactory(
        amplitude_parameter=amplitude_parameter,
        mass_split=neutrino_mass_splits,
        require_nonlinear_pk=require_nonlinear_pk,
        num_neutrino_masses=num_neutrino_masses,
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
    with pytest.raises(
        ValueError,
        match=(
            "CAMB extra parameters, CAMB halo model sampling, "
            "and multiple CAMB instances are only compatible "
            "with the PURE_CCL_MODE creation mode when using "
            "the BOLTZMANN_CAMB transfer function."
        ),
    ):
        CCLFactory(camb_extra_params={"dark_energy_model": "Im not a valid value"})


def test_camb_extra_params_hmcode_logT_with_old_mead() -> None:
    for halofit_version in ["mead", "mead2015", "mead2016"]:
        with pytest.raises(
            ValueError,
            match=(
                f"HMCode_logT_AGN is not available for "
                f"halofit_version={halofit_version}"
            ),
        ):
            CAMBExtraParams(
                halofit_version=halofit_version,
                HMCode_logT_AGN=7.8,
            )


def test_camb_extra_params_hmcode_A_eta_with_mead2020_feedback() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "HMCode_A_baryon and HMCode_eta_baryon are only available for "
            "halofit_version in \\(mead, mead2015, mead2016\\)"
        ),
    ):
        CAMBExtraParams(
            halofit_version="mead2020_feedback",
            HMCode_A_baryon=3.13,
        )

    with pytest.raises(
        ValueError,
        match=(
            "HMCode_A_baryon and HMCode_eta_baryon are only available for "
            "halofit_version in \\(mead, mead2015, mead2016\\)"
        ),
    ):
        CAMBExtraParams(
            halofit_version="mead2020_feedback",
            HMCode_eta_baryon=0.603,
        )


def test_camb_extra_params_hmcode_with_unknown_halofit() -> None:
    unknown_versions = ["invalid_version", "halofit2", "peacock"]
    for version in unknown_versions:
        with pytest.raises(
            ValueError,
            match=(
                f"HMCode parameters are not compatible with "
                f"halofit_version={version}"
            ),
        ):
            CAMBExtraParams(
                halofit_version=version,
                HMCode_A_baryon=3.13,
            )


def test_camb_extra_params_hmcode_with_no_halofit() -> None:
    with pytest.raises(
        ValueError,
        match="Value error, HMCode_logT_AGN is not available for halofit_version",
    ):
        CAMBExtraParams(HMCode_logT_AGN=7.8)


def test_camb_extra_params_hmcode_with_another_halofit() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Value error, HMCode parameters are not compatible with "
            "halofit_version=another"
        ),
    ):
        CAMBExtraParams(halofit_version="another", HMCode_logT_AGN=7.8)
    with pytest.raises(
        ValueError,
        match=(
            "Value error, HMCode parameters are not compatible with "
            "halofit_version=another"
        ),
    ):
        CAMBExtraParams(halofit_version="another", HMCode_A_baryon=3.13)
    with pytest.raises(
        ValueError,
        match=(
            "Value error, HMCode parameters are not compatible with "
            "halofit_version=another"
        ),
    ):
        CAMBExtraParams(halofit_version="another", HMCode_eta_baryon=0.603)

    params = CAMBExtraParams(halofit_version="another")
    assert params is not None
    assert isinstance(params, CAMBExtraParams)
    assert params.HMCode_logT_AGN is None
    assert params.HMCode_A_baryon is None
    assert params.HMCode_eta_baryon is None
    assert params.halofit_version == "another"


def test_camb_extra_params_valid_mead() -> None:
    for halofit_version in [None, "mead", "mead2015", "mead2016"]:
        params = CAMBExtraParams(
            halofit_version=halofit_version,
            HMCode_A_baryon=3.13,
            HMCode_eta_baryon=0.603,
        )
        assert params.HMCode_A_baryon == 3.13
        assert params.HMCode_eta_baryon == 0.603


def test_camb_extra_params_valid_mead2020_feedback() -> None:
    params = CAMBExtraParams(
        halofit_version="mead2020_feedback",
        HMCode_logT_AGN=7.8,
    )
    assert params.HMCode_logT_AGN == 7.8


def test_camb_extra_params_valid_no_hmcode() -> None:
    for halofit_version in [None, "mead", "mead2015", "mead2016", "mead2020_feedback"]:
        params = CAMBExtraParams(halofit_version=halofit_version)
        assert params.halofit_version == halofit_version


def test_empty_camb_extra_params() -> None:
    params = CAMBExtraParams()
    assert params.halofit_version is None
    assert params.is_mead()
    assert not params.is_mead2020_feedback()


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
        ValueError, match="Calculator Mode can only be used with the DEFAULT creation."
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

    with pytest.raises(ValueError, match=r"Parameters have not been updated yet\."):
        mu_sigma_model.create()


def test_bad_configuration() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"To sample over the halo model, you must include camb_extra_parameters\."
        ),
    ):
        CCLFactory(
            creation_mode=CCLCreationMode.PURE_CCL_MODE,
            use_camb_hm_sampling=True,
            camb_extra_params=None,
        )


def test_hm_sampling_misconfiguration() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "CAMB extra parameters, CAMB halo model sampling, "
            "and multiple CAMB instances are only compatible "
            "with the PURE_CCL_MODE creation mode when using "
            "the BOLTZMANN_CAMB transfer function."
        ),
    ):
        _ = CCLFactory(use_camb_hm_sampling=True, camb_extra_params=CAMBExtraParams())


@pytest.mark.parametrize(
    "halofit_version", ["mead", "mead2015", "mead2016", "mead2020_feedback"]
)
def test_hm_sampling_configuration(halofit_version: str) -> None:
    factory = CCLFactory(
        creation_mode=CCLCreationMode.PURE_CCL_MODE,
        use_camb_hm_sampling=True,
        camb_extra_params=CAMBExtraParams(halofit_version=halofit_version),
    )
    assert factory.camb_extra_params is not None
    camb_extra_params: CAMBExtraParams = factory.camb_extra_params
    assert isinstance(camb_extra_params, CAMBExtraParams)
    # There is something here that is confusing pylint
    # pylint: disable=no-member
    is_mead = camb_extra_params.is_mead()
    is_mead2020_feedback = factory.camb_extra_params.is_mead2020_feedback()
    if is_mead:
        assert camb_extra_params.HMCode_A_baryon is None
        assert camb_extra_params.HMCode_eta_baryon is None
    if is_mead2020_feedback:
        assert factory.HMCode_logT_AGN is None

    # Update the factory to make it have default values
    params = get_default_params_map(factory)
    factory.update(params)
    if is_mead:
        assert camb_extra_params.HMCode_A_baryon == 3.13
        assert camb_extra_params.HMCode_eta_baryon == 0.603
    if is_mead2020_feedback:
        assert camb_extra_params.HMCode_logT_AGN == 7.8
    # pylint: enable=no-member


@pytest.mark.parametrize(
    "creation_mode,transfer_function,expected",
    [
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.BOLTZMANN_CAMB,
            True,
        ),
        (CCLCreationMode.PURE_CCL_MODE, CCLPureModeTransferFunction.BBKS, False),
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.BOLTZMANN_CLASS,
            False,
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.EISENSTEIN_HU,
            False,
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.EISENSTEIN_HU_NOWIGGLES,
            False,
        ),
        (CCLCreationMode.DEFAULT, CCLPureModeTransferFunction.BOLTZMANN_CAMB, False),
        (
            CCLCreationMode.MU_SIGMA_ISITGR,
            CCLPureModeTransferFunction.BOLTZMANN_CAMB,
            False,
        ),
    ],
)
def test_ccl_factory_using_camb(creation_mode, transfer_function, expected):
    factory = CCLFactory(
        creation_mode=creation_mode,
        pure_ccl_transfer_function=transfer_function,
    )
    assert factory.using_camb() is expected


@pytest.mark.parametrize(
    "creation_mode,transfer_function,nonlinear_pk,matter_pk_str",
    [
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.BOLTZMANN_CAMB,
            True,
            "halofit",
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.BBKS,
            True,
            "halofit",
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.BOLTZMANN_CLASS,
            True,
            "halofit",
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.EISENSTEIN_HU,
            True,
            "halofit",
        ),
        (
            CCLCreationMode.PURE_CCL_MODE,
            CCLPureModeTransferFunction.EISENSTEIN_HU_NOWIGGLES,
            True,
            "halofit",
        ),
        (
            CCLCreationMode.DEFAULT,
            CCLPureModeTransferFunction.BOLTZMANN_CAMB,
            True,
            "halofit",
        ),
        (
            CCLCreationMode.MU_SIGMA_ISITGR,
            CCLPureModeTransferFunction.BOLTZMANN_CAMB,
            True,
            "halofit",
        ),
    ],
)
def test_ccl_factory_ccl_powerspectra(
    creation_mode, transfer_function, nonlinear_pk, matter_pk_str
):
    factory = CCLFactory(
        creation_mode=creation_mode,
        pure_ccl_transfer_function=transfer_function,
        require_nonlinear_pk=nonlinear_pk,
    )
    cosmo_params = get_default_params_map(factory)
    factory.update(cosmo_params)
    cosmo = factory.create()
    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)
    assert cosmo.to_dict()["matter_power_spectrum"] == matter_pk_str


@pytest.mark.parametrize(
    "m_nu,As_sigma8,mass_split",
    it.product(
        [0.0, 0.1, 0.2], [(2.0e-9, None), (None, 0.81)], list(NeutrinoMassSplits)
    ),
)
def test_ccl_factory_parameters_passthrough_all(m_nu, As_sigma8, mass_split) -> None:
    """Ensure all cosmology parameters provided by the user are passed through
    unchanged to the created CCL cosmology when using the CCLFactory/ModelingTools
    plumbing.

    This test builds a reference `pyccl.Cosmology` directly from the parameter
    dictionary and compares each parameter against the `pyccl.Cosmology` created
    by `ModelingTools` using a `CCLFactory`. The `mass_split='equal'` mode is
    specifically exercised since that was the reported failing case.
    """
    As, sigma8 = As_sigma8
    cosmo_dict = {
        "Omega_c": 0.2906682,
        "Omega_b": 0.04575,
        "h": 0.6714,
        "n_s": 0.9493,
        "Neff": 3.044,
        "m_nu": m_nu,
        "w0": -1.0,
        "wa": 0.0,
        "T_CMB": 2.7255,
        "Omega_k": 0.0,
    }
    amplitude_parameter: PoweSpecAmplitudeParameter | None = None
    if As is not None:
        cosmo_dict["A_s"] = As
        amplitude_parameter = PoweSpecAmplitudeParameter.AS
    if sigma8 is not None:
        cosmo_dict["sigma8"] = sigma8
        amplitude_parameter = PoweSpecAmplitudeParameter.SIGMA8

    m_nu_is_list = _nu_mass_is_list(mass_split)

    assert amplitude_parameter is not None
    tools = ModelingTools(
        ccl_factory=CCLFactory(
            amplitude_parameter=amplitude_parameter,
            mass_split=mass_split,
            num_neutrino_masses=(3 if m_nu_is_list else None),
        )
    )

    if m_nu_is_list:
        cosmo_dict["m_nu"] = 0.01
        cosmo_dict["m_nu_2"] = 0.02
        cosmo_dict["m_nu_3"] = 0.03

    tools.update(ParamsMap(cosmo_dict))
    tools.prepare()

    if m_nu_is_list:
        cosmo_dict["m_nu"] = [0.01, 0.02, 0.03]
        del cosmo_dict["m_nu_2"]
        del cosmo_dict["m_nu_3"]

    # Reference cosmology built directly with pyccl
    ccl_cosmo = pyccl.Cosmology(**cosmo_dict, mass_split=mass_split)
    tools_cosmo = tools.ccl_cosmo

    assert isinstance(tools_cosmo, pyccl.Cosmology)
    for key in cosmo_dict:
        expected = ccl_cosmo[key]
        got = tools_cosmo[key]
        assert_allclose(expected, got, rtol=1e-12, atol=0)
