"""
Tests for deprecated module imports.
"""

import pytest


def test_import_gauss_family():
    with pytest.deprecated_call():
        # pylint: disable=import-outside-toplevel
        from firecrown.likelihood.gauss_family import gauss_family

    assert gauss_family is not None
