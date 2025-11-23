"""Integration tests for the TwoPointTheory class using real Source instances."""

from firecrown.models.two_point import TwoPointTheory
from firecrown.parameters import ParamsMap
from firecrown.likelihood import Source


class TestSource(Source):
    """A minimal concrete Source implementation for testing."""

    def __init__(self, sacc_tracer: str):
        super().__init__(sacc_tracer)

    def read_systematics(self, sacc_data) -> None:
        pass

    def _read(self, sacc_data) -> None:
        pass

    def _update_source(self, params) -> None:
        pass

    def get_scale(self) -> float:
        return 1.0

    def create_tracers(self, tools):
        return []


def test_two_point_theory_update_integration():
    """Integration test: update method updates real sources."""
    source1 = TestSource("tracer_1")
    source2 = TestSource("tracer_2")
    sources = (source1, source2)

    theory = TwoPointTheory(
        sacc_data_type="galaxy_density_cl",
        sources=sources,
    )

    params = ParamsMap({})
    theory.update(params)

    assert source1.is_updated()
    assert source2.is_updated()


def test_two_point_theory_reset_integration():
    """Integration test: reset method resets real sources."""
    source1 = TestSource("tracer_1")
    source2 = TestSource("tracer_2")
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

    # Call the protected _reset method which is what gets invoked
    theory._reset()  # pylint: disable=protected-access

    assert not source1.is_updated()
    assert not source2.is_updated()
