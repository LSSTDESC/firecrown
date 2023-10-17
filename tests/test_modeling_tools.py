"""
Tests for the module firecrown.modeling_tools
"""
import pytest
import pyccl
from firecrown.modeling_tools import ModelingTools


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


def test_no_adding_pk_twice(dummy_powerspectrum: pyccl.Pk2D):
    tools = ModelingTools()
    tools.add_pk("silly", dummy_powerspectrum)
    assert tools.powerspectra["silly"] == dummy_powerspectrum
    with pytest.raises(KeyError):
        tools.add_pk("silly", dummy_powerspectrum)
