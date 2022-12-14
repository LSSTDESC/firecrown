import cosmosis.datablock
import pytest
from firecrown.connector.cosmosis.likelihood import FirecrownLikelihood


@pytest.fixture(name="minimal_module_config")
def fixture_minimal_module_config():
    """Return a minimal CosmoSIS datablock.
    It contains only the module's filename.
    This is the minimal possible configuration."""
    block = cosmosis.datablock.DataBlock()
    block.put_string(
        "module_options", "likelihood_source", "tests/likelihood/lkdir/lkscript.py"
    )
    return block


def test_parameterless_module_construction(minimal_module_config):
    """Make sure we can create a CosmoSIS likelihood modules that does not
    introduce any new parameters."""
    module = FirecrownLikelihood(minimal_module_config)
    assert module.sampling_sections == []
