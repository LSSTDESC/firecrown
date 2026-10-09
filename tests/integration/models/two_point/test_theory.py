"""Integration tests for the TwoPointTheory class using real Source instances."""

import pytest
import sacc

from firecrown.likelihood import Source, Tracer
from firecrown.modeling_tools import ModelingTools
from firecrown.models.two_point import TwoPointTheory
from firecrown.updatable import ParamsMap


class FakeSource(Source):
    """A minimal concrete Source implementation for testing."""

    def read_systematics(self, sacc_data: sacc.Sacc) -> None:
        pass

    def _read(self, sacc_data: sacc.Sacc) -> None:
        pass

    def _update_source(self, params: ParamsMap) -> None:
        pass

    def get_scale(self) -> float:
        return 1.0

    def create_tracers(self, tools: ModelingTools) -> tuple[list[Tracer], object]:
        raise NotImplementedError("FakeSource only supports update and reset tests.")


@pytest.mark.integration
def test_two_point_theory_update_integration() -> None:
    """Integration test: update method updates real sources."""
    source1 = FakeSource("tracer_1")
    source2 = FakeSource("tracer_2")
    sources = (source1, source2)

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=sources,
    )

    params = ParamsMap({})
    theory.update(params)

    assert source1.is_updated()
    assert source2.is_updated()


@pytest.mark.integration
def test_two_point_theory_reset_integration() -> None:
    """Integration test: reset method resets real sources."""
    source1 = FakeSource("tracer_1")
    source2 = FakeSource("tracer_2")
    sources = (source1, source2)

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=sources,
    )

    # First update to set them as updated
    params = ParamsMap({})
    theory.update(params)
    assert source1.is_updated()
    assert source2.is_updated()

    # Ensure reset makes them not updated
    theory.reset()

    assert not source1.is_updated()
    assert not source2.is_updated()
