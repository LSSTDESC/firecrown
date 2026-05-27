"""Tests for the deprecated firecrown.likelihood.gaussian module.

This module verifies that the deprecated firecrown.likelihood.gaussian module
correctly re-exports :class:`ConstGaussian` from its current location
(:mod:`firecrown.likelihood._gaussian`) with appropriate deprecation warnings.

Note: Only the first test checks for warnings, as Python's import system
caches modules and won't re-execute the module-level code that emits the
warning on subsequent imports.
"""

import sys
import pytest


def test_gaussian_deprecation_warning():
    """Importing firecrown.likelihood.gaussian emits a DeprecationWarning.

    This must be the first test that imports the module, as subsequent imports
    from the same Python process will use the cached module.
    """
    if "firecrown.likelihood.gaussian" in sys.modules:
        del sys.modules["firecrown.likelihood.gaussian"]

    with pytest.warns(
        DeprecationWarning,
        match="firecrown.likelihood.gaussian is deprecated",
    ):
        # pylint: disable=import-outside-toplevel,unused-import
        import firecrown.likelihood.gaussian  # noqa: F401


def test_const_gaussian_importable():
    """ConstGaussian can be imported from the deprecated module."""
    # pylint: disable=import-outside-toplevel
    from firecrown.likelihood.gaussian import ConstGaussian

    assert ConstGaussian is not None


def test_const_gaussian_identical_to_new_location():
    """ConstGaussian from deprecated module is the same object as from _gaussian."""
    # pylint: disable=import-outside-toplevel
    from firecrown.likelihood.gaussian import ConstGaussian as OldConstGaussian
    from firecrown.likelihood._gaussian import ConstGaussian

    assert OldConstGaussian is ConstGaussian


def test_const_gaussian_in_all():
    """ConstGaussian is listed in __all__ of the deprecated module."""
    # pylint: disable=import-outside-toplevel
    import firecrown.likelihood.gaussian as gaussian_module

    assert "ConstGaussian" in gaussian_module.__all__
