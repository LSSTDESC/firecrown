"""
Tests for the module firecrown.modeling_tools
"""

import pytest
import pyccl
from firecrown.modeling_tools import ModelingTools, PowerspectrumModifier
from firecrown.parameters import ParamsMap


@pytest.fixture(name="dummy_powerspectrum")
def make_dummy_powerspectrum() -> pyccl.Pk2D:
    """Create an empty power spectrum. This is the only type we can create
    without supplying a cosmology."""
    return pyccl.Pk2D.__new__(pyccl.Pk2D)


def test_default_constructed_state():
    tools = ModelingTools()
    # Default constructed state is pretty barren...
    assert tools.ccl_cosmo is None
    assert tools.pt_calculator is None
    assert tools.cluster_abundance is None
    assert len(tools.powerspectra) == 0


def test_default_constructed_no_tools():
    tools = ModelingTools()
    with pytest.raises(RuntimeError):
        _ = tools.get_pk("nonesuch")


def test_adding_pk_and_getting(dummy_powerspectrum: pyccl.Pk2D):
    tools = ModelingTools()
    tools.add_pk("silly", dummy_powerspectrum)
    cosmo = pyccl.CosmologyVanillaLCDM()
    tools.update(ParamsMap())
    tools.prepare(cosmo)

    assert tools.get_pk("silly") == dummy_powerspectrum


def test_no_adding_pk_twice(dummy_powerspectrum: pyccl.Pk2D):
    tools = ModelingTools()
    tools.add_pk("silly", dummy_powerspectrum)
    assert tools.powerspectra["silly"] == dummy_powerspectrum
    with pytest.raises(KeyError):
        tools.add_pk("silly", dummy_powerspectrum)


def test_modeling_tool_prepare_without_update():
    tools = ModelingTools()
    cosmo = pyccl.CosmologyVanillaLCDM()
    with pytest.raises(RuntimeError, match="ModelingTools has not been updated."):
        tools.prepare(cosmo)


def test_modeling_tool_preparing_twice():
    tools = ModelingTools()
    cosmo = pyccl.CosmologyVanillaLCDM()
    tools.update(ParamsMap())
    tools.prepare(cosmo)
    with pytest.raises(RuntimeError, match="ModelingTools has already been prepared"):
        tools.prepare(cosmo)


def test_modeling_tool_wrongly_setting_ccl_cosmo():
    tools = ModelingTools()
    cosmo = pyccl.CosmologyVanillaLCDM()
    tools.update(ParamsMap())
    tools.ccl_cosmo = cosmo
    with pytest.raises(RuntimeError, match="Cosmology has already been set"):
        tools.prepare(cosmo)


def test_modeling_tools_get_cosmo_without_setting():
    tools = ModelingTools()
    with pytest.raises(RuntimeError, match="Cosmology has not been set"):
        tools.get_ccl_cosmology()


def test_modeling_tools_get_pt_calculator_without_setting():
    tools = ModelingTools()
    with pytest.raises(RuntimeError, match="A PT calculator has not been set"):
        tools.get_pt_calculator()


class DummyPowerspectrumModifier(PowerspectrumModifier):
    """Dummy power spectrum modifier for testing."""

    def __init__(self):
        super().__init__()
        self.name = "dummy:dummy"

    def compute_p_of_k_z(self, tools: ModelingTools) -> pyccl.Pk2D:
        assert tools is not None
        assert isinstance(tools, ModelingTools)
        return pyccl.Pk2D.__new__(pyccl.Pk2D)


def test_modeling_tools_add_pk_modifiers():
    tools = ModelingTools(pk_modifiers=[DummyPowerspectrumModifier()])
    cosmo = pyccl.CosmologyVanillaLCDM()
    tools.update(ParamsMap())
    tools.prepare(cosmo)

    assert tools.get_pk("dummy:dummy") is not None
    assert isinstance(tools.get_pk("dummy:dummy"), pyccl.Pk2D)
