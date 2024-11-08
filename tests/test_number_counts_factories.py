"""Tests for factories for number counts systematics."""

import pytest
import firecrown.likelihood.number_counts as nc
import firecrown.utils as fcutils


@pytest.fixture(name="nc_factory")
def fixture_nc_factory() -> nc.NumberCountsFactory:
    """Fixture for the NumberCountsFactory class.
    The resulting factory will creates
    """
    number_counts_yaml = """
    per_bin_systematics:
    - type: PhotoZShiftFactory
    global_systematics: []
    """
    return fcutils.base_model_from_yaml(nc.NumberCountsFactory, number_counts_yaml)


def test_photozshift_not_applied_to_photometic_measurement(
):
    sys = nc.PhotoZShiftFactory(sacc_tracer="lens0")
    assert sys.parameter_prefix == "lens0"
