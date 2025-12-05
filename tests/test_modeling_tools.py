"""
Tests for the module firecrown.modeling_tools
"""

import pytest
import pyccl
from firecrown.updatable import get_default_params_map
from firecrown.modeling_tools import ModelingTools, PowerspectrumModifier
from firecrown.models.cluster import ClusterAbundance, ClusterDeltaSigma


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
    assert tools.cluster_abundance is None
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
    """Test get_hm_calculator when hm_calculator is set."""
    # Create a mock halo model calculator with required parameters
    mass_function = pyccl.halos.MassFuncTinker10()
    halo_bias = pyccl.halos.HaloBiasTinker10()
    hm_calculator = pyccl.halos.HMCalculator(
        mass_function=mass_function, halo_bias=halo_bias
    )
    tools = ModelingTools(hm_calculator=hm_calculator)

    # This should return the hm_calculator successfully
    result = tools.get_hm_calculator()
    assert result is hm_calculator


def test_modeling_tools_get_cM_relation_without_setting() -> None:
    tools = ModelingTools()
    with pytest.raises(
        RuntimeError, match="A concentration-mass relation has not been set"
    ):
        tools.get_cM_relation()


def test_modeling_tools_get_cM_relation_with_setting() -> None:
    """Test get_cM_relation when cM_relation is set."""
    cM_relation = "duffy2008"
    tools = ModelingTools(cM_relation=cM_relation)

    # This should return the cM_relation successfully
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


def test_get_pk_fallback_to_ccl_cosmology() -> None:
    """Test get_pk falls back to ccl_cosmo.get_nonlin_power.

    When a power spectrum is not in the local table, get_pk should fall back
    to requesting it from the CCL cosmology object.
    """
    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    # Request a power spectrum that's not in the local table
    # CCL cosmology should provide 'delta_matter:delta_matter'
    pk = tools.get_pk("delta_matter:delta_matter")
    assert pk is not None
    assert isinstance(pk, pyccl.Pk2D)


def test_has_pk_returns_true_for_existing_pk(dummy_powerspectrum: pyccl.Pk2D) -> None:
    """Test has_pk returns True for existing power spectrum."""
    tools = ModelingTools()
    tools.add_pk("test_pk", dummy_powerspectrum)
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    assert tools.has_pk("test_pk") is True


def test_has_pk_returns_false_for_nonexistent_pk() -> None:
    """Test has_pk returns False for nonexistent power spectrum.

    When the power spectrum doesn't exist in the local table and CCL raises
    KeyError, has_pk should catch it and return False.
    """
    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    # Request a power spectrum that CCL doesn't have
    # This should raise KeyError internally, which has_pk catches
    assert tools.has_pk("nonexistent:invalid") is False


def test_get_ccl_cosmology_success() -> None:
    """Test get_ccl_cosmology returns cosmology object successfully."""
    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    cosmo = tools.get_ccl_cosmology()
    assert cosmo is not None
    assert isinstance(cosmo, pyccl.Cosmology)
    assert cosmo is tools.ccl_cosmo


def test_get_pt_calculator_success() -> None:
    """Test get_pt_calculator returns PT calculator successfully."""
    # Create a real PT calculator
    pt_calculator = pyccl.nl_pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=False,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=20,
    )

    tools = ModelingTools(pt_calculator=pt_calculator)
    params = get_default_params_map(tools)
    tools.update(params)

    # Get the PT calculator before prepare to avoid update_ingredients call
    result = tools.get_pt_calculator()
    assert result is not None
    assert result is pt_calculator


def test_reset_method() -> None:
    """Test _reset method clears ccl_cosmo, powerspectra, and _prepared flag."""
    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    # Verify tools are prepared and have data
    assert tools.ccl_cosmo is not None
    assert tools._prepared is True  # pylint: disable=protected-access

    # Call _reset
    tools._reset()  # pylint: disable=protected-access

    # Verify everything is reset
    assert tools.ccl_cosmo is None
    assert not tools.powerspectra
    assert tools._prepared is False  # pylint: disable=protected-access


def test_prepare_with_pt_calculator_updates_ingredients() -> None:
    """Test prepare() calls update_ingredients on pt_calculator when set."""
    # Create a real PT calculator
    pt_calculator = pyccl.nl_pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=False,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=20,
    )

    tools = ModelingTools(pt_calculator=pt_calculator)
    params = get_default_params_map(tools)
    tools.update(params)

    # Before prepare, the PT calculator should exist but not have cosmology set
    assert tools.pt_calculator is not None

    # Call prepare - this should call update_ingredients on the PT calculator
    tools.prepare()

    # After prepare, verify the cosmology was set and PT calculator was updated
    assert tools.ccl_cosmo is not None


def test_prepare_with_cluster_abundance_updates_ingredients() -> None:
    """Test prepare() calls update_ingredients on cluster_abundance when set."""
    # Create a real ClusterAbundance object
    hmf = pyccl.halos.MassFuncTinker08(mass_def="200c")
    min_mass, max_mass = 13.0, 16.0
    min_z, max_z = 0.2, 0.8
    cluster_abundance = ClusterAbundance((min_mass, max_mass), (min_z, max_z), hmf)

    tools = ModelingTools(cluster_abundance=cluster_abundance)
    params = get_default_params_map(tools)
    tools.update(params)

    # Before prepare, cluster_abundance should not have cosmology set
    assert cluster_abundance.cosmo is None

    # Call prepare - this should call update_ingredients on cluster_abundance
    tools.prepare()

    # After prepare, verify the cosmology was set on cluster_abundance
    assert tools.ccl_cosmo is not None
    assert cluster_abundance.cosmo is not None
    assert cluster_abundance.cosmo is tools.ccl_cosmo


def test_prepare_with_cluster_deltasigma_updates_ingredients() -> None:
    """Test prepare() calls update_ingredients on cluster_deltasigma when set."""
    # Create a real ClusterDeltaSigma object
    hmf = pyccl.halos.MassFuncTinker08(mass_def="200c")
    min_mass, max_mass = 13.0, 16.0
    min_z, max_z = 0.2, 0.8
    cluster_deltasigma = ClusterDeltaSigma(
        (min_mass, max_mass), (min_z, max_z), hmf, conc_parameter=True
    )

    tools = ModelingTools(cluster_deltasigma=cluster_deltasigma)
    params = get_default_params_map(tools)
    tools.update(params)

    # Before prepare, cluster_deltasigma should not have cosmology set
    assert cluster_deltasigma.cosmo is None

    # Call prepare - this should call update_ingredients on cluster_deltasigma
    tools.prepare()

    # After prepare, verify the cosmology was set on cluster_deltasigma
    assert tools.ccl_cosmo is not None
    assert cluster_deltasigma.cosmo is not None
    assert cluster_deltasigma.cosmo is tools.ccl_cosmo
