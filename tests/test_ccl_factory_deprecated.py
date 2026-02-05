"""Tests for the deprecated firecrown.ccl_factory module.

This module tests that the deprecated firecrown.ccl_factory package:
1. Emits appropriate deprecation warnings when imported
2. Maintains backward compatibility by re-exporting all items
3. Re-exported items are identical to those in firecrown.modeling_tools

Note: Only the first test checks for warnings, as Python's import system
caches modules and won't re-execute the module-level code that emits the warning.
"""

import sys
import pytest


def test_ccl_factory_deprecation_warning():
    """Test that importing ccl_factory emits DeprecationWarning.

    This must be the first test that imports the module, as subsequent imports
    from the same Python process will use the cached module.
    """
    # Remove the module if it's already imported
    if "firecrown.ccl_factory" in sys.modules:
        del sys.modules["firecrown.ccl_factory"]

    with pytest.warns(
        DeprecationWarning,
        match="firecrown.ccl_factory is deprecated and will be removed",
    ):
        # pylint: disable=import-outside-toplevel,unused-import
        import firecrown.ccl_factory  # noqa: F401


def test_ccl_factory_deprecation_warning_content():
    """Test the deprecation warning mentions modeling_tools.

    This test relies on module caching, so it won't see the warning.
    We just verify the module is importable.
    """
    # pylint: disable=import-outside-toplevel,unused-import
    import firecrown.ccl_factory  # noqa: F401

    # If we get here without error, the import worked


def test_all_items_importable():
    """Test that all __all__ items can be imported from deprecated package."""
    # pylint: disable=import-outside-toplevel
    from firecrown.ccl_factory import (
        Background,
        CAMBExtraParams,
        CCLCalculatorArgs,
        CCLCreationMode,
        CCLFactory,
        CCLPureModeTransferFunction,
        CCLSplineParams,
        MuSigmaModel,
        PoweSpecAmplitudeParameter,
        PowerSpec,
    )

    # Verify all imports succeeded and are not None
    assert Background is not None
    assert CAMBExtraParams is not None
    assert CCLCalculatorArgs is not None
    assert CCLCreationMode is not None
    assert CCLFactory is not None
    assert CCLPureModeTransferFunction is not None
    assert CCLSplineParams is not None
    assert MuSigmaModel is not None
    assert PoweSpecAmplitudeParameter is not None
    assert PowerSpec is not None


# pylint: disable=too-many-locals
def test_items_identical_to_new_location():
    """Test that imported items are the same objects as in modeling_tools."""
    # pylint: disable=import-outside-toplevel
    # Import from both locations
    from firecrown.ccl_factory import (
        Background as OldBackground,
        CAMBExtraParams as OldCAMBExtraParams,
        CCLCalculatorArgs as OldCCLCalculatorArgs,
        CCLCreationMode as OldCCLCreationMode,
        CCLFactory as OldCCLFactory,
        CCLPureModeTransferFunction as OldCCLPureModeTransferFunction,
        CCLSplineParams as OldCCLSplineParams,
        MuSigmaModel as OldMuSigmaModel,
        PoweSpecAmplitudeParameter as OldPoweSpecAmplitudeParameter,
        PowerSpec as OldPowerSpec,
    )

    from firecrown.modeling_tools import (
        Background,
        CAMBExtraParams,
        CCLCalculatorArgs,
        CCLCreationMode,
        CCLFactory,
        CCLPureModeTransferFunction,
        CCLSplineParams,
        MuSigmaModel,
        PoweSpecAmplitudeParameter,
        PowerSpec,
    )

    # Verify they are the same objects (identity, not just equality)
    assert OldBackground is Background
    assert OldCAMBExtraParams is CAMBExtraParams
    assert OldCCLCalculatorArgs is CCLCalculatorArgs
    assert OldCCLCreationMode is CCLCreationMode
    assert OldCCLFactory is CCLFactory
    assert OldCCLPureModeTransferFunction is CCLPureModeTransferFunction
    assert OldCCLSplineParams is CCLSplineParams
    assert OldMuSigmaModel is MuSigmaModel
    assert OldPoweSpecAmplitudeParameter is PoweSpecAmplitudeParameter
    assert OldPowerSpec is PowerSpec


def test_module_import_as_alias():
    """Test importing the module with an alias."""
    # pylint: disable=import-outside-toplevel
    import firecrown.ccl_factory as fac

    # Verify we can access items through the alias
    assert hasattr(fac, "CCLFactory")
    assert hasattr(fac, "PoweSpecAmplitudeParameter")
    assert hasattr(fac, "CCLCreationMode")


def test_ccl_factory_all_list():
    """Test that __all__ is preserved in deprecated module."""
    # pylint: disable=import-outside-toplevel
    import firecrown.ccl_factory

    expected_all = [
        "PowerSpec",
        "Background",
        "CCLCalculatorArgs",
        "PoweSpecAmplitudeParameter",
        "CCLCreationMode",
        "CCLPureModeTransferFunction",
        "MuSigmaModel",
        "CAMBExtraParams",
        "CCLSplineParams",
        "CCLFactory",
    ]

    assert set(firecrown.ccl_factory.__all__) == set(expected_all)


def test_ccl_factory_functional():
    """Test that CCLFactory from deprecated module is functional."""
    # pylint: disable=import-outside-toplevel
    from firecrown.ccl_factory import CCLFactory, PoweSpecAmplitudeParameter

    # Create a CCLFactory instance
    factory = CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)

    # Verify basic functionality
    assert factory.amplitude_parameter == PoweSpecAmplitudeParameter.SIGMA8
    assert factory is not None
