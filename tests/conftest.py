"""
pytest configuration additions.
"""
import pytest


def pytest_addoption(parser):
    """Add handling of firecrown-specific options for the pytest test runner.

    --runslow: used to run tests marked as slow, which are otherwise not run.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Add new markers that can be set on pytest tests.

    Use the marker `slow` for any test that takes more than a second to run.
    Tests marked as `slow` are not run unless the user requests them by specifying
    the --runslow flag to the pytest program.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Apply our special markers and option handling for pytest."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
