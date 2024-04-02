"""
Test that the version string of the firecrown module is appropriately set.
Note that this test intentionally has to be adjusted for each new release of
firecrown.
"""

import firecrown


def test_version():
    assert firecrown.__version__ == "1.7.2"
