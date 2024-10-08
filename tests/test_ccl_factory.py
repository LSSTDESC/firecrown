"""Test the CCLFactory object."""

import numpy as np
import pytest
import pyccl
from pyccl.neutrinos import NeutrinoMassSplits

from firecrown.ccl_factory import (
    CCLFactory,
    PoweSpecAmplitudeParameter,
    CCLCalculatorArgs,
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
