"""Unit testing for firecrown's CosmoSIS module.

As a unit test, what this can test is very limited.
This test do not invoke the `cosmosis` executable.
"""
from os.path import expandvars
import yaml
import pytest
import numpy as np
from cosmosis.datablock import DataBlock, option_section, names as section_names

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.connector.cosmosis.likelihood import FirecrownLikelihood, extract_section


@pytest.fixture(name="minimal_module_config")
def fixture_minimal_module_config() -> DataBlock:
    """Return a minimal CosmoSIS datablock.
    It contains only the module's filename.
    This is the minimal possible configuration."""
    block = DataBlock()
    block.put_string(
        option_section, "likelihood_source", "tests/likelihood/lkdir/lkscript.py"
    )
    return block


@pytest.fixture(name="defective_module_config")
def fixture_defective_module_config() -> DataBlock:
    """Return a CosmoSIS datablock that lacks the required
    parameter to configure a ParameterizedLikelihood."""
    block = DataBlock()
    block.put_string(
        option_section,
        "likelihood_source",
        expandvars("${FIRECROWN_DIR}/tests/likelihood/lkdir/lk_needing_param.py"),
    )
    return block


@pytest.fixture(name="minimal_config")
def fixture_minimal_config() -> DataBlock:
    result = DataBlock()
    result.put_string(
        option_section,
        "likelihood_source",
        expandvars("${FIRECROWN_DIR}/tests/likelihood/lkdir/lkscript.py"),
    )
    return result


@pytest.fixture(name="config_with_derived_parameters")
def fixture_config_with_derived_parameters() -> DataBlock:
    result = DataBlock()
    result.put_string(
        option_section,
        "Likelihood_source",
        expandvars("${FIRECROWN_DIR}/tests/likelihood/lkdir/lk_derived_parameter.py"),
    )
    result.put_double("sample_parameters_sections", "derived_param0", 12.5)
    return result


@pytest.fixture(name="minimal_firecrown_mod")
def fixture_minimal_firecrown_mod(minimal_config: DataBlock) -> FirecrownLikelihood:
    return FirecrownLikelihood(minimal_config)


@pytest.fixture(name="firecrown_mod_with_derived_parameters")
def fixture_firecrown_mod_with_derived_parameters(
    config_with_derived_parameters: DataBlock,
) -> FirecrownLikelihood:
    return FirecrownLikelihood(config_with_derived_parameters)


@pytest.fixture(name="sample_with_cosmo")
def fixture_sample_with_cosmo() -> DataBlock:
    """Return a DataBlock that contains some cosmological parameters."""
    result = DataBlock()
    params = {
        "h0": 0.83,
        "omega_b": 0.22,
        "omega_c": 0.120,
        "omega_k": 0.0,
        "omega_nu": 0.0,
        "w": -1.0,
        "wa": 0.0,
    }
    for name, val in params.items():
        result.put_double("cosmological_parameters", name, val)
    return result


@pytest.fixture(name="minimal_sample")
def fixture_minimal_sample(sample_with_cosmo: DataBlock) -> DataBlock:
    with open("tests/distances.yml", encoding="utf-8") as stream:
        rawdata = yaml.load(stream, yaml.CLoader)
    sample = sample_with_cosmo
    for section_name, section_data in rawdata.items():
        for parameter_name, value in section_data.items():
            sample.put(section_name, parameter_name, np.array(value))
    return sample


def test_extract_section_gets_named_parameters(defective_module_config: DataBlock):
    params = extract_section(defective_module_config, "module_options")
    assert isinstance(params, NamedParameters)
    assert params.get_string("likelihood_source") == expandvars(
        "${FIRECROWN_DIR}/tests/likelihood/lkdir/lk_needing_param.py"
    )


def test_extract_section_raises_on_missing_section(defective_module_config: DataBlock):
    with pytest.raises(RuntimeError, match="Datablock section `xxx' does not exist"):
        _ = extract_section(defective_module_config, "xxx")


def test_parameterless_module_construction(minimal_module_config):
    """Make sure we can create a CosmoSIS likelihood modules that does not
    introduce any new parameters."""
    module = FirecrownLikelihood(minimal_module_config)
    assert module.sampling_sections == []


def test_missing_required_parameter(defective_module_config):
    """Make sure that a missing required parameter entails the expected
    failure."""
    with pytest.raises(KeyError):
        _ = FirecrownLikelihood(defective_module_config)


def test_initialize_minimal_module(minimal_firecrown_mod: FirecrownLikelihood):
    assert isinstance(minimal_firecrown_mod, FirecrownLikelihood)


def test_execute_missing_cosmological_parameters(
    minimal_firecrown_mod: FirecrownLikelihood,
):
    no_cosmo_params = DataBlock()
    with pytest.raises(
        RuntimeError,
        match="Datablock section `cosmological_parameters' does not exist.",
    ):
        _ = minimal_firecrown_mod.execute(no_cosmo_params)


def test_execute_with_cosmo(
    minimal_firecrown_mod: FirecrownLikelihood, minimal_sample: DataBlock
):
    assert minimal_firecrown_mod.execute(minimal_sample) == 0
    assert minimal_sample[section_names.likelihoods, "firecrown_like"] == -3.0


def test_execute_with_derived_parameters(
    firecrown_mod_with_derived_parameters: FirecrownLikelihood,
    minimal_sample: DataBlock,
):
    assert firecrown_mod_with_derived_parameters.execute(minimal_sample) == 0
    assert minimal_sample.get_double("derived_section", "derived_param0") == 1.0
