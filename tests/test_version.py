"""
Test that the version string of the firecrown module is appropriately set.
This test validates version format and components, so it doesn't need
to be updated with each release.
"""

import re

import firecrown
import firecrown.version


def test_version_format():
    """Test that version follows semantic versioning format.

    This test validates the version format rather than a specific value,
    so it doesn't need to be updated with each release.
    """
    # Match semantic versioning: MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCHaX
    pattern = r"^\d+\.\d+\.\d+(?:a\d+)?$"
    assert re.match(
        pattern, firecrown.__version__
    ), f"Version {firecrown.__version__} doesn't follow semantic versioning"


def test_version_components():
    """Test that version components are accessible and valid."""
    # Verify the version components are accessible and reasonable
    assert firecrown.version.FIRECROWN_MAJOR >= 1
    assert firecrown.version.FIRECROWN_MINOR >= 0
    assert firecrown.version.FIRECROWN_PATCH is not None

    # Verify __version__ matches the components
    expected = (
        f"{firecrown.version.FIRECROWN_MAJOR}."
        f"{firecrown.version.FIRECROWN_MINOR}."
        f"{firecrown.version.FIRECROWN_PATCH}"
    )
    assert firecrown.__version__ == expected
