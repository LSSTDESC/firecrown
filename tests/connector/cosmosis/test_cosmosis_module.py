"""Unit testing for firecrown's CosmoSIS module.

As a unit test, what this can test is very limited.
This test do not invoke the `cosmosis` executable.
"""


from cosmosis.datablock import DataBlock, option_section
import pytest
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
        "tests/likelihood/lkdir/lk_needing_param.py",
    )
    return block


@pytest.fixture(name="minimal_config")
def fixture_minimal_config() -> DataBlock:
    result = DataBlock()
    result.put_string(
        option_section, "likelihood_source", "tests/likelihood/lkdir/lkscript.py"
    )
    return result


@pytest.fixture(name="firecrown_mod")
def fixture_firecrown_mod(minimal_config: DataBlock) -> FirecrownLikelihood:
    return FirecrownLikelihood(minimal_config)


def test_extract_section_gets_named_parameters(defective_module_config: DataBlock):
    params = extract_section(defective_module_config, "module_options")
    assert isinstance(params, NamedParameters)
    assert (
        params.get_string("likelihood_source")
        == "tests/likelihood/lkdir/lk_needing_param.py"
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


def test_initialize_minimal_module(firecrown_mod: FirecrownLikelihood):
    assert isinstance(firecrown_mod, FirecrownLikelihood)


def test_execute_missing_cosmological_parameters(firecrown_mod: FirecrownLikelihood):
    no_cosmo_params = DataBlock()
    with pytest.raises(
        RuntimeError,
        match="Datablock section " "`cosmological_parameters' does " "not exist.",
    ):
        _ = firecrown_mod.execute(no_cosmo_params)
