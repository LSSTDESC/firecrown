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


@pytest.fixture(name="nc_sys_factory")
def fixture_nc_sys_factory() -> nc.NumberCountsSystematicFactory:
    """Fixture for the NumberCountsSystematicFactory class."""
    # yaml = """
    # global_systematics: []
    # per_bin_systematics:
    # - type: PhotoZShiftFactory
    # """

    return nc.PhotoZShiftFactory()


def test_photozshift_not_applied_to_photometic_measurement():

    factory = nc.PhotoZShiftFactory()
    assert factory is not None
