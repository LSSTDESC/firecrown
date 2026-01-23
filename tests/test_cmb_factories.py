"""Tests for the CMB convergence module."""

from unittest import mock
import pytest
import numpy as np
import sacc


from firecrown.likelihood._cmb import (
    CMBConvergence,
    CMBConvergenceFactory,
    CMBConvergenceArgs,
)
from firecrown.metadata_types import TomographicBin, CMB
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import ParamsMap


def test_cmb_convergence_args():
    """Test CMBConvergenceArgs creation and properties."""
    args = CMBConvergenceArgs(scale=2.0, field="delta_matter", z_source=1090.0)

    assert args.scale == 2.0
    assert args.field == "delta_matter"
    assert args.z_source == 1090.0


def test_cmb_convergence_init():
    """Test CMBConvergence initialization."""
    cmb_conv = CMBConvergence(sacc_tracer="cmb", scale=1.5, z_source=1090.0)

    assert cmb_conv.sacc_tracer == "cmb"
    assert cmb_conv.scale == 1.5
    assert cmb_conv.z_source == 1090.0
    assert cmb_conv.current_tracer_args is None


def test_cmb_convergence_create_ready():
    """Test CMBConvergence.create_ready method."""
    cmb_conv = CMBConvergence.create_ready(
        sacc_tracer="cmb", scale=2.0, z_source=1090.0
    )

    assert cmb_conv.sacc_tracer == "cmb"
    assert cmb_conv.scale == 2.0
    assert cmb_conv.z_source == 1090.0
    assert cmb_conv.tracer_args.scale == 2.0
    assert cmb_conv.tracer_args.field == "delta_matter"
    assert cmb_conv.tracer_args.z_source == 1090.0


def test_cmb_convergence_read_systematics():
    """Test that read_systematics does nothing for CMB sources."""
    cmb_conv = CMBConvergence(sacc_tracer="cmb")

    # Create a mock sacc data object
    mock_sacc = sacc.Sacc()

    # This should not raise any errors and should do nothing
    cmb_conv.read_systematics(mock_sacc)


def test_cmb_convergence_read(sacc_galaxy_cells_src0_src0):
    """Test CMBConvergence _read method."""
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0

    cmb_conv = CMBConvergence.create_ready(sacc_tracer="cmb", scale=1.0)

    with mock.patch.object(sacc_data, "get_tracer", return_value=mock.Mock()):
        cmb_conv._read(sacc_data)  # pylint: disable=protected-access

    assert cmb_conv.tracer_args.scale == 1.0
    assert cmb_conv.tracer_args.field == "delta_matter"


def test_cmb_convergence_read_missing_tracer_args():
    """Test that _read properly initializes tracer_args."""
    cmb_conv = CMBConvergence(sacc_tracer="cmb", scale=2.0, z_source=1090.0)

    # Verify tracer_args doesn't exist initially
    assert not hasattr(cmb_conv, "tracer_args")

    # Create a SACC data object
    sacc_data = sacc.Sacc()

    with mock.patch.object(sacc_data, "get_tracer", return_value=mock.Mock()):
        cmb_conv._read(sacc_data)  # pylint: disable=protected-access

    # Verify tracer_args was created correctly
    assert hasattr(cmb_conv, "tracer_args")
    assert cmb_conv.tracer_args.scale == 2.0
    assert cmb_conv.tracer_args.z_source == 1090.0
    assert cmb_conv.tracer_args.field == "delta_matter"


def test_cmb_convergence_create_tracers(tools_with_vanilla_cosmology: ModelingTools):
    """Test CMBConvergence create_tracers method."""
    cmb_conv = CMBConvergence.create_ready(
        sacc_tracer="cmb", scale=1.0, z_source=1100.0
    )

    tracers, tracer_args = cmb_conv.create_tracers(tools_with_vanilla_cosmology)

    assert len(tracers) == 1
    assert tracers[0].tracer_name == "cmb_convergence"
    assert tracers[0].field == "delta_matter"
    assert tracer_args.scale == 1.0
    assert tracer_args.z_source == 1100.0
    assert cmb_conv.current_tracer_args == tracer_args


def test_cmb_convergence_get_scale():
    """Test CMBConvergence get_scale method."""
    cmb_conv = CMBConvergence.create_ready(sacc_tracer="cmb", scale=2.5)

    # Before create_tracers, current_tracer_args should be None
    with pytest.raises(RuntimeError, match="current_tracer_args is not initialized"):
        cmb_conv.get_scale()

    # Set current_tracer_args manually for testing
    cmb_conv.current_tracer_args = CMBConvergenceArgs(scale=2.5)

    assert cmb_conv.get_scale() == 2.5


def test_cmb_convergence_factory_init():
    """Test CMBConvergenceFactory initialization."""
    factory = CMBConvergenceFactory(z_source=1090.0, scale=2.0)

    assert factory.z_source == 1090.0
    assert factory.scale == 2.0
    assert not factory._cache  # pylint: disable=protected-access


def test_cmb_convergence_factory_create():
    """Test CMBConvergenceFactory create method."""
    factory = CMBConvergenceFactory(z_source=1090.0, scale=1.5)

    # Create a mock TomographicBin with CMB measurements
    mock_zdist = TomographicBin(
        bin_name="cmb_bin",
        z=np.linspace(0, 2, 100),
        dndz=np.ones(100),
        measurements={CMB.CONVERGENCE},
    )

    cmb_conv = factory.create(mock_zdist)

    assert cmb_conv.sacc_tracer == "cmb_bin"
    assert cmb_conv.scale == 1.5
    assert cmb_conv.z_source == 1090.0


def test_cmb_convergence_factory_create_caching():
    """Test that CMBConvergenceFactory caches created objects."""
    factory = CMBConvergenceFactory()

    mock_zdist = TomographicBin(
        bin_name="cmb_bin",
        z=np.linspace(0, 2, 100),
        dndz=np.ones(100),
        measurements={CMB.CONVERGENCE},
    )

    cmb_conv1 = factory.create(mock_zdist)
    cmb_conv2 = factory.create(mock_zdist)

    # Should return the same cached object
    assert cmb_conv1 is cmb_conv2


def test_cmb_convergence_factory_create_from_metadata_only():
    """Test CMBConvergenceFactory create_from_metadata_only method."""
    factory = CMBConvergenceFactory(z_source=1100.0, scale=1.0)

    cmb_conv = factory.create_from_metadata_only("cmb_tracer")

    assert cmb_conv.sacc_tracer == "cmb_tracer"
    assert cmb_conv.scale == 1.0
    assert cmb_conv.z_source == 1100.0


def test_cmb_convergence_factory_create_from_metadata_caching():
    """Test that create_from_metadata_only also uses caching."""
    factory = CMBConvergenceFactory()

    cmb_conv1 = factory.create_from_metadata_only("cmb_tracer")
    cmb_conv2 = factory.create_from_metadata_only("cmb_tracer")

    # Should return the same cached object
    assert cmb_conv1 is cmb_conv2


def test_cmb_convergence_no_systematics():
    """Test that CMB sources have no systematics."""
    cmb_conv = CMBConvergence(sacc_tracer="cmb")

    # CMB sources should not have systematics attribute
    assert not hasattr(cmb_conv, "systematics")


def test_cmb_convergence_update_params():
    """Test that CMBConvergence can handle parameter updates."""
    cmb_conv = CMBConvergence.create_ready(sacc_tracer="cmb")

    # Should not raise any errors even with parameters
    # (CMB sources don't use parameters but should handle the interface)
    params = ParamsMap()
    cmb_conv.update(params)


def test_cmb_convergence_args_immutable():
    """Test that CMBConvergenceArgs is immutable."""
    args = CMBConvergenceArgs(scale=1.0)

    # Should not be able to modify after creation
    with pytest.raises(Exception):  # Could be AttributeError or FrozenInstanceError
        args.scale = 2.0  # type: ignore[misc]


def test_cmb_convergence_factory_different_params():
    """Test factory with different parameters creates different objects."""
    factory1 = CMBConvergenceFactory(z_source=1090.0, scale=1.0)
    factory2 = CMBConvergenceFactory(z_source=1100.0, scale=2.0)

    mock_zdist = TomographicBin(
        bin_name="cmb_bin",
        z=np.linspace(0, 2, 100),
        dndz=np.ones(100),
        measurements={CMB.CONVERGENCE},
    )

    cmb_conv1 = factory1.create(mock_zdist)
    cmb_conv2 = factory2.create(mock_zdist)

    assert cmb_conv1.z_source == 1090.0
    assert cmb_conv1.scale == 1.0
    assert cmb_conv2.z_source == 1100.0
    assert cmb_conv2.scale == 2.0


@pytest.mark.parametrize("z_source", [1090.0, 1100.0, 1110.0])
def test_cmb_convergence_different_z_source(
    z_source: float, tools_with_vanilla_cosmology: ModelingTools
):
    """Test CMBConvergence with different z_source values."""
    cmb_conv = CMBConvergence.create_ready(sacc_tracer="cmb", z_source=z_source)

    tracers, tracer_args = cmb_conv.create_tracers(tools_with_vanilla_cosmology)

    assert tracer_args.z_source == z_source
    assert len(tracers) == 1


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0])
def test_cmb_convergence_different_scales(scale: float):
    """Test CMBConvergence with different scale values."""
    cmb_conv = CMBConvergence.create_ready(sacc_tracer="cmb", scale=scale)

    assert cmb_conv.scale == scale
    assert cmb_conv.tracer_args.scale == scale
