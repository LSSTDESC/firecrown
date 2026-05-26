"""
Test that the version string of the firecrown module is appropriately set.
This test validates version format and components, so it doesn't need
to be updated with each release.
"""

import re

import firecrown

SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:a\d+)?$")


def test_version_format():
    """Test that version follows semantic versioning format.

    This test validates the version format rather than a specific value,
    so it doesn't need to be updated with each release.
    """
    assert hasattr(
        firecrown, "__version__"
    ), "firecrown module has no __version__ attribute"

    # Extract just the release version (strip .dev, .post, and other local parts)
    version = firecrown.__version__
    match = re.match(r"^(\d+\.\d+\.\d+(?:a\d+)?)", version)
    assert match, f"Could not parse version {version}"
    base_version = match.group(1)

    assert SEMVER_PATTERN.match(
        base_version
    ), f"Version {base_version} doesn't follow semantic versioning"
