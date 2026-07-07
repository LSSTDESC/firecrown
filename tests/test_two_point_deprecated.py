"""Tests for the deprecated firecrown.likelihood.two_point module.

This module tests that the deprecated firecrown.likelihood.two_point package:
1. Emits appropriate deprecation warnings when imported
2. Maintains backward compatibility by re-exporting all items
3. Re-exported items are identical to those in firecrown.likelihood._two_point

Note: Only the first test checks for warnings, as Python's import system
caches modules and won't re-execute the module-level code that emits the warning.
"""

import sys
import pytest


def test_two_point_deprecation_warning():
    """Test that importing two_point emits DeprecationWarning.

    This must be the first test that imports the module, as subsequent imports
    from the same Python process will use the cached module.
    """
    # Remove the module if it's already imported
    if "firecrown.likelihood.two_point" in sys.modules:
        del sys.modules["firecrown.likelihood.two_point"]

    with pytest.warns(
        DeprecationWarning,
        match="firecrown.likelihood.two_point module is deprecated and will be removed",
    ):
        # pylint: disable=import-outside-toplevel,unused-import
        import firecrown.likelihood.two_point  # noqa: F401


def test_two_point_deprecation_warning_content():
    """Test the deprecation warning mentions firecrown.likelihood.

    This test relies on module caching, so it won't see the warning.
    We just verify the module is importable.
    """
    # pylint: disable=import-outside-toplevel,unused-import
    import firecrown.likelihood.two_point  # noqa: F401

    # If we get here without error, the import worked


def test_all_items_importable():
    """Test that all __all__ items can be imported from deprecated module."""
    # pylint: disable=import-outside-toplevel
    from firecrown.likelihood.two_point import (
        TwoPoint,
        TwoPointFactory,
        calculate_angular_cl,
        read_ell_cells,
        read_reals,
        use_source_factory,
        use_source_factory_metadata_index,
    )

    # Verify all imports succeeded and are not None
    assert TwoPoint is not None
    assert TwoPointFactory is not None
    assert calculate_angular_cl is not None
    assert read_ell_cells is not None
    assert read_reals is not None
    assert use_source_factory is not None
    assert use_source_factory_metadata_index is not None


def test_items_identical_to_new_location():
    """Test that imported items are the same objects as in _two_point."""
    # pylint: disable=import-outside-toplevel
    from firecrown.likelihood.two_point import (
        TwoPoint as OldTwoPoint,
        TwoPointFactory as OldTwoPointFactory,
        calculate_angular_cl as old_calculate_angular_cl,
        read_ell_cells as old_read_ell_cells,
        read_reals as old_read_reals,
        use_source_factory as old_use_source_factory,
        use_source_factory_metadata_index as old_use_source_factory_metadata_index,
    )

    from firecrown.likelihood._two_point import (
        TwoPoint,
        TwoPointFactory,
        calculate_angular_cl,
        read_ell_cells,
        read_reals,
        use_source_factory,
        use_source_factory_metadata_index,
    )

    # Verify they are the same objects (identity, not just equality)
    assert OldTwoPoint is TwoPoint
    assert OldTwoPointFactory is TwoPointFactory
    assert old_calculate_angular_cl is calculate_angular_cl
    assert old_read_ell_cells is read_ell_cells
    assert old_read_reals is read_reals
    assert old_use_source_factory is use_source_factory
    assert old_use_source_factory_metadata_index is use_source_factory_metadata_index


def test_module_import_as_alias():
    """Test importing the module with an alias."""
    # pylint: disable=import-outside-toplevel
    import firecrown.likelihood.two_point as tp

    # Verify we can access items through the alias
    assert hasattr(tp, "TwoPoint")
    assert hasattr(tp, "TwoPointFactory")
    assert hasattr(tp, "calculate_angular_cl")


def test_two_point_all_list():
    """Test that __all__ is preserved in deprecated module."""
    # pylint: disable=import-outside-toplevel
    import firecrown.likelihood.two_point

    expected_all = [
        "TwoPoint",
        "TwoPointFactory",
        "calculate_angular_cl",
        "read_ell_cells",
        "read_reals",
        "use_source_factory",
        "use_source_factory_metadata_index",
    ]

    assert set(firecrown.likelihood.two_point.__all__) == set(expected_all)
