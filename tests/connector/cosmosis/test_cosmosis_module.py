"""Unit testing for firecrown's CosmoSIS module.

As a unit test, what this can test is very limited.
This test do not invoke the `cosmosis` executable.
"""


from cosmosis.datablock import DataBlock
import pytest
from firecrown.connector.cosmosis.likelihood import FirecrownLikelihood


@pytest.fixture(name="minimal_module_config")
def fixture_minimal_module_config() -> DataBlock:
    """Return a minimal CosmoSIS datablock.
    It contains only the module's filename.
    This is the minimal possible configuration."""
    block = DataBlock()
    block.put_string(
        "module_options", "likelihood_source", "tests/likelihood/lkdir/lkscript.py"
    )
    return block


@pytest.fixture(name="defective_module_config")
def fixture_defective_module_config() -> DataBlock:
    """Return a CosmoSIS datablock that lacks the required
    parameter to configure a ParameterizedLikelihood."""
    block = DataBlock()
    block.put_string(
        "module_options",
        "likelihood_source",
        "tests/likelihood/lkdir/lk_needing_param.py",
    )
    return block


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
