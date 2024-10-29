"""Test the CCLFactory object."""

import numpy as np
import pytest
import pyccl
import pyccl.modified_gravity
from pyccl.neutrinos import NeutrinoMassSplits

from firecrown.ccl_factory import (
    CAMBExtraParams,
    CCLCalculatorArgs,
    CCLCreationMode,
    CCLFactory,
    MuSigmaModel,
    PoweSpecAmplitudeParameter,
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


Z_ARRAY = np.linspace(0.0, 5.0, 100)
A_ARRAY = 1.0 / (1.0 + np.flip(Z_ARRAY))
K_ARRAY = np.geomspace(1.0e-5, 10.0, 100)
A_GRID, K_GRID = np.meshgrid(A_ARRAY, K_ARRAY, indexing="ij")

CHI_ARRAY = np.linspace(100.0, 0.0, 100)
H_OVER_H0_ARRAY = np.linspace(1.0, 100.0, 100)
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


def test_ccl_factory_simple(
    amplitude_parameter: PoweSpecAmplitudeParameter,
    neutrino_mass_splits: NeutrinoMassSplits,
    require_nonlinear_pk: bool,
    ccl_creation_mode: CCLCreationMode,
    camb_extra_params: CAMBExtraParams | None,
) -> None:
    ccl_factory = CCLFactory(
        amplitude_parameter=amplitude_parameter,
        mass_split=neutrino_mass_splits,
        require_nonlinear_pk=require_nonlinear_pk,
        creation_mode=ccl_creation_mode,
        camb_extra_params=camb_extra_params,
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


def test_ccl_factory_ccl_args(
    amplitude_parameter: PoweSpecAmplitudeParameter,
    neutrino_mass_splits: NeutrinoMassSplits,
    require_nonlinear_pk: bool,
    calculator_args: CCLCalculatorArgs,
) -> None:
    ccl_factory = CCLFactory(
        amplitude_parameter=amplitude_parameter,
        mass_split=neutrino_mass_splits,
        require_nonlinear_pk=require_nonlinear_pk,
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
    ccl_factory = CCLFactory(neutrino_mass_splits=neutrino_mass_splits)

    assert ccl_factory is not None

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
        match=".*Invalid value for PoweSpecAmplitudeParameter: Im not a valid value.*",
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
        match=".*Invalid value for CCLCreationMode: Im not a valid value.*",
    ):
        CCLFactory(creation_mode="Im not a valid value")


def test_ccl_factory_from_dict() -> None:
    ccl_factory_dict = {
        "amplitude_parameter": PoweSpecAmplitudeParameter.SIGMA8,
        "mass_split": NeutrinoMassSplits.EQUAL,
        "require_nonlinear_pk": True,
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
