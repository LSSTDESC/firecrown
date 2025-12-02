"""
Tests for the module firecrown.modeling_tools
"""

import pytest
import pyccl
from firecrown.updatable import get_default_params_map
from firecrown.modeling_tools import ModelingTools, PowerspectrumModifier


@pytest.fixture(name="dummy_powerspectrum", scope="session")
def make_dummy_powerspectrum() -> pyccl.Pk2D:
    """Create an empty power spectrum. This is the only type we can create
    without supplying a cosmology."""
    return pyccl.Pk2D.__new__(pyccl.Pk2D)


def test_default_constructed_state() -> None:
    tools = ModelingTools()
    # Default constructed state is pretty barren...
    assert tools.ccl_cosmo is None
    assert tools.pt_calculator is None
    assert len(tools.powerspectra) == 0


def test_default_constructed_no_tools() -> None:
    tools = ModelingTools()
    with pytest.raises(RuntimeError):
        _ = tools.get_pk("nonesuch")


def test_adding_pk_and_getting(dummy_powerspectrum: pyccl.Pk2D) -> None:
    tools = ModelingTools()
    tools.add_pk("silly", dummy_powerspectrum)
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    assert tools.get_pk("silly") == dummy_powerspectrum


def test_no_adding_pk_twice(dummy_powerspectrum: pyccl.Pk2D) -> None:
    tools = ModelingTools()
    tools.add_pk("silly", dummy_powerspectrum)
    assert tools.powerspectra["silly"] == dummy_powerspectrum
    with pytest.raises(KeyError):
        tools.add_pk("silly", dummy_powerspectrum)


def test_modeling_tool_prepare_without_update() -> None:
    tools = ModelingTools()
    with pytest.raises(RuntimeError, match="ModelingTools has not been updated."):
        tools.prepare()


def test_modeling_tool_preparing_twice() -> None:
    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()
    with pytest.raises(RuntimeError, match="ModelingTools has already been prepared"):
        tools.prepare()


def test_modeling_tool_wrongly_setting_ccl_cosmo() -> None:
    tools = ModelingTools()
    cosmo = pyccl.CosmologyVanillaLCDM()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.ccl_cosmo = cosmo
    with pytest.raises(RuntimeError, match="Cosmology has already been set"):
        tools.prepare()


def test_modeling_tools_get_cosmo_without_setting() -> None:
    tools = ModelingTools()
    with pytest.raises(RuntimeError, match="Cosmology has not been set"):
        tools.get_ccl_cosmology()


def test_modeling_tools_get_pt_calculator_without_setting() -> None:
    tools = ModelingTools()
    with pytest.raises(RuntimeError, match="A PT calculator has not been set"):
        tools.get_pt_calculator()


def test_modeling_tools_get_hm_calculator_without_setting() -> None:
    tools = ModelingTools()
    with pytest.raises(RuntimeError, match="A Halo Model calculator has not been set"):
        tools.get_hm_calculator()


def test_modeling_tools_get_hm_calculator_with_setting() -> None:
    """Test get_hm_calculator when hm_calculator is set (covers line 152)."""
    # Create a mock halo model calculator with required parameters
    mass_function = pyccl.halos.MassFuncTinker10()
    halo_bias = pyccl.halos.HaloBiasTinker10()
    hm_calculator = pyccl.halos.HMCalculator(
        mass_function=mass_function, halo_bias=halo_bias
    )
    tools = ModelingTools(hm_calculator=hm_calculator)

    # This should return the hm_calculator successfully (line 152)
    result = tools.get_hm_calculator()
    assert result is hm_calculator


def test_modeling_tools_get_cM_relation_without_setting() -> None:
    tools = ModelingTools()
    with pytest.raises(
        RuntimeError, match="A concentration-mass relation has not been set"
    ):
        tools.get_cM_relation()


def test_modeling_tools_get_cM_relation_with_setting() -> None:
    """Test get_cM_relation when cM_relation is set (covers line 158)."""
    cM_relation = "duffy2008"
    tools = ModelingTools(cM_relation=cM_relation)

    # This should return the cM_relation successfully (line 158)
    result = tools.get_cM_relation()
    assert result == cM_relation


class DummyPowerspectrumModifier(PowerspectrumModifier):
    """Dummy power spectrum modifier for testing."""

    def __init__(self):
        super().__init__()
        self.name = "dummy:dummy"

    def compute_p_of_k_z(self, tools: ModelingTools) -> pyccl.Pk2D:
        assert tools is not None
        assert isinstance(tools, ModelingTools)
        return pyccl.Pk2D.__new__(pyccl.Pk2D)


def test_modeling_tools_add_pk_modifiers() -> None:
    tools = ModelingTools(pk_modifiers=[DummyPowerspectrumModifier()])
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    assert tools.get_pk("dummy:dummy") is not None
    assert isinstance(tools.get_pk("dummy:dummy"), pyccl.Pk2D)
