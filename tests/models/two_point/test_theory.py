"""Unit tests for the theory module (TwoPointTheory class)."""

from unittest.mock import Mock
import numpy as np
import pytest
import sacc

from firecrown.models.two_point import TwoPointTheory
from firecrown.generators import LogLinearElls, EllOrThetaConfig
from firecrown.metadata_types import TracerNames
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from firecrown.utils import ClIntegrationOptions
from firecrown.models.two_point import ApplyInterpolationWhen
from firecrown.likelihood import Source


@pytest.fixture
def mock_source():
    """Create a mock source that behaves like a real Updatable Source."""
    source = Mock(spec=Source)
    source.sacc_tracer = "tracer_1"
    source._updated = False
    source.is_updated = Mock(return_value=False)

    def update_side_effect(params, updated_record=None):
        if source._updated:
            return  # Mimic Updatable: do nothing if already updated
        source._updated = True
        source.is_updated.return_value = True

    def reset_side_effect():
        source._updated = False
        source.is_updated.return_value = False

    source.update = Mock(side_effect=update_side_effect)
    source.reset = Mock(side_effect=reset_side_effect)
    return source


@pytest.fixture
def mock_source_pair(mock_source):
    """Create a pair of mock sources."""
    source2 = Mock(spec=Source)
    source2.sacc_tracer = "tracer_2"
    source2._updated = False
    source2.is_updated = Mock(return_value=False)

    def update_side_effect2(params, updated_record=None):
        if source2._updated:
            return
        source2._updated = True
        source2.is_updated.return_value = True

    def reset_side_effect2():
        source2._updated = False
        source2.is_updated.return_value = False

    source2.update = Mock(side_effect=update_side_effect2)
    source2.reset = Mock(side_effect=reset_side_effect2)
    return (mock_source, source2)


class TestSource(Source):
    """A minimal concrete Source implementation for testing."""

    def __init__(self, sacc_tracer: str):
        super().__init__(sacc_tracer)

    def read_systematics(self, sacc_data: sacc.Sacc) -> None:
        pass

    def _read(self, sacc_data: sacc.Sacc) -> None:
        pass

    def _update_source(self, params: ParamsMap) -> None:
        pass

    def get_scale(self) -> float:
        return 1.0

    def create_tracers(self, tools: ModelingTools):
        return []


@pytest.fixture
def source_pair():
    """Create a pair of concrete TestSource instances."""
    return (TestSource("tracer_1"), TestSource("tracer_2"))


def test_two_point_theory_initialization():
    """Test basic TwoPointTheory initialization."""
    source1 = Mock()
    source2 = Mock()

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source1, source2),
    )

    assert theory.sacc_data_type == "galaxy_density_cl"
    assert theory.sources == (source1, source2)
    assert theory.ells is None
    assert theory.thetas is None
    assert theory.window is None
    assert not theory.cells


def test_two_point_theory_with_custom_interp_ells():
    """Test TwoPointTheory with custom interpolation ells generator."""
    source1 = Mock()
    source2 = Mock()
    interp_ells_gen = LogLinearElls(n_log=10, midpoint=30, maximum=1000)

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source1, source2),
        interp_ells_gen=interp_ells_gen,
    )

    assert theory.interp_ells_gen == interp_ells_gen
    assert theory.ells_for_xi is not None
    assert len(theory.ells_for_xi) > 0


def test_two_point_theory_with_ell_or_theta_config():
    """Test TwoPointTheory initialization with ell_or_theta configuration."""
    source1 = Mock()
    source2 = Mock()
    ell_or_theta = EllOrThetaConfig(  # type: ignore[typeddict-item]
        ells=np.array([10, 20, 30], dtype=np.int64), ell_type="window"
    )

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source1, source2),
        ell_or_theta=ell_or_theta,
    )

    assert theory.ell_or_theta_config == ell_or_theta


def test_two_point_theory_with_tracers():
    """Test TwoPointTheory initialization with explicit tracer names."""
    source1 = Mock()
    source2 = Mock()
    tracers = TracerNames("tracer_1", "tracer_2")

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source1, source2),
        tracers=tracers,
    )

    assert theory.sacc_tracers == tracers


def test_two_point_theory_with_int_options():
    """Test TwoPointTheory with integration options."""
    source1 = Mock()
    source2 = Mock()
    int_options = ClIntegrationOptions(
        method="fkem_auto",  # type: ignore[arg-type]
        limber_method="gsl_qag_quad",  # type: ignore[arg-type]
    )

    theory = TwoPointTheory(
        sacc_data_type="galaxy_shear_cl_ee",
        sources=(source1, source2),
        int_options=int_options,
    )

    assert theory.int_options == int_options


def test_two_point_theory_with_apply_interp():
    """Test TwoPointTheory with custom apply_interp flag."""
    source1 = Mock()
    source2 = Mock()

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source1, source2),
        apply_interp=ApplyInterpolationWhen.ALL,
    )

    assert theory.apply_interp == ApplyInterpolationWhen.ALL


def test_two_point_theory_source0_property(mock_source_pair):
    """Test source0 property returns first source."""
    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=mock_source_pair,
    )

    assert theory.source0 == mock_source_pair[0]


def test_two_point_theory_source1_property(mock_source_pair):
    """Test source1 property returns second source."""
    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=mock_source_pair,
    )

    assert theory.source1 == mock_source_pair[1]


def test_two_point_theory_update(mock_source_pair):
    """Test update method calls update on sources."""
    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=mock_source_pair,
    )

    params = ParamsMap({})
    theory.update(params)

    # Check that sources are marked as updated
    assert mock_source_pair[0].is_updated()
    assert mock_source_pair[1].is_updated()


def test_two_point_theory_reset(mock_source_pair):
    """Test reset method calls reset on sources."""
    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=mock_source_pair,
    )

    # First update to set them as updated
    params = ParamsMap({})
    theory.update(params)
    assert mock_source_pair[0].is_updated()
    assert mock_source_pair[1].is_updated()

    # Call the protected _reset method which is what gets invoked
    theory._reset()  # pylint: disable=protected-access

    assert not mock_source_pair[0].is_updated()
    assert not mock_source_pair[1].is_updated()


def test_two_point_theory_initialize_sources_different_sources():
    """Test initialize_sources with two different sources."""
    source1 = Mock()
    source1.sacc_tracer = "tracer_1"
    source2 = Mock()
    source2.sacc_tracer = "tracer_2"

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source1, source2),
    )

    sacc_data = Mock(spec=sacc.Sacc)
    theory.initialize_sources(sacc_data)

    source1.read.assert_called_once_with(sacc_data)
    source2.read.assert_called_once_with(sacc_data)
    assert theory.sacc_tracers == TracerNames("tracer_1", "tracer_2")


def test_two_point_theory_initialize_sources_same_source():
    """Test initialize_sources with the same source for both."""
    source = Mock()
    source.sacc_tracer = "tracer_1"

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source, source),
    )

    sacc_data = Mock(spec=sacc.Sacc)
    theory.initialize_sources(sacc_data)

    # Should only call read once since both sources are the same
    source.read.assert_called_once_with(sacc_data)
    assert theory.sacc_tracers == TracerNames("tracer_1", "tracer_1")


def test_two_point_theory_get_tracers_and_scales_different_sources():
    """Test get_tracers_and_scales with different sources."""
    source1 = Mock()
    source1_tracers = [Mock(), Mock()]
    source1.get_tracers.return_value = source1_tracers
    source1.get_scale.return_value = 1.5

    source2 = Mock()
    source2_tracers = [Mock()]
    source2.get_tracers.return_value = source2_tracers
    source2.get_scale.return_value = 2.0

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source1, source2),
    )

    mock_tools = Mock(spec=ModelingTools)
    tracers0, scale0, tracers1, scale1 = theory.get_tracers_and_scales(mock_tools)

    assert tracers0 == source1_tracers
    assert scale0 == 1.5
    assert tracers1 == source2_tracers
    assert scale1 == 2.0


def test_two_point_theory_get_tracers_and_scales_same_source():
    """Test get_tracers_and_scales with the same source."""
    source = Mock()
    source_tracers = [Mock(), Mock()]
    source.get_tracers.return_value = source_tracers
    source.get_scale.return_value = 1.5

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source, source),
    )

    mock_tools = Mock(spec=ModelingTools)
    tracers0, scale0, tracers1, scale1 = theory.get_tracers_and_scales(mock_tools)

    # Should only call get_tracers and get_scale once
    source.get_tracers.assert_called_once_with(mock_tools)
    source.get_scale.assert_called_once()

    assert tracers0 == source_tracers
    assert tracers1 == source_tracers
    assert scale0 == 1.5
    assert scale1 == 1.5


def test_two_point_theory_generate_ells_for_interpolation():
    """Test generate_ells_for_interpolation method."""
    source1 = Mock()
    source2 = Mock()

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=(source1, source2),
        interp_ells_gen=LogLinearElls(n_log=5, midpoint=30, maximum=1000),
    )

    # Set ells to simulate data
    theory.ells = np.array([10, 50, 100, 200, 500], dtype=np.int64)

    result = theory.generate_ells_for_interpolation()

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) > 0
    assert result[0] >= 10  # Should start at or after min_ell
    assert result[-1] <= 500  # Should end at or before max_ell
