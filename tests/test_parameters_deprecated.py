"""Tests for the deprecated firecrown.parameters module.

This module verifies that the deprecated firecrown.parameters module correctly
re-exports all functionality from firecrown.updatable with appropriate
deprecation warnings.
"""

import sys
import warnings


def test_parameters_module_emits_deprecation_warning():
    """Importing firecrown.parameters should emit a DeprecationWarning."""
    # Remove the module from sys.modules if it's already imported
    # so we can import it fresh and catch the warning
    if "firecrown.parameters" in sys.modules:
        del sys.modules["firecrown.parameters"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Import should trigger warning
        # pylint: disable=unused-import,import-outside-toplevel
        import firecrown.parameters  # noqa: F401

        # Should have at least one warning (there might be multiple due to
        # imports within the module)
        assert len(w) >= 1
        # First warning should be a DeprecationWarning
        assert issubclass(w[0].category, DeprecationWarning)
        # Check message content
        assert "firecrown.parameters module is deprecated" in str(w[0].message)
        assert "firecrown.updatable" in str(w[0].message)


def test_parameters_exports_all_names():
    """All expected names should be available from firecrown.parameters."""
    # pylint: disable=import-outside-toplevel
    import firecrown.parameters as params

    expected_names = [
        "DerivedParameter",
        "DerivedParameterCollection",
        "InternalParameter",
        "ParamsMap",
        "RequiredParameters",
        "SamplerParameter",
        "handle_unused_params",
        "parameter_get_full_name",
        "register_new_updatable_parameter",
    ]

    for name in expected_names:
        assert hasattr(params, name), f"Missing export: {name}"


def test_parameters_exports_are_same_as_updatable():
    """Exported names should be the same objects as in firecrown.updatable."""
    # pylint: disable=import-outside-toplevel
    import firecrown.parameters as params
    from firecrown import updatable

    names_to_check = [
        "DerivedParameter",
        "DerivedParameterCollection",
        "InternalParameter",
        "ParamsMap",
        "RequiredParameters",
        "SamplerParameter",
        "handle_unused_params",
        "parameter_get_full_name",
        "register_new_updatable_parameter",
    ]

    for name in names_to_check:
        params_obj = getattr(params, name)
        updatable_obj = getattr(updatable, name)
        assert (
            params_obj is updatable_obj
        ), f"{name} from parameters is not the same object as from updatable"


def test_params_map_from_deprecated_module():
    """Ensure ParamsMap imported from deprecated module works correctly."""
    # pylint: disable=import-outside-toplevel
    from firecrown.parameters import ParamsMap

    params = ParamsMap({"a": 1.0, "b": 2.0})
    assert params.get("a") == 1.0
    assert params.get("b") == 2.0


def test_required_parameters_from_deprecated_module():
    """Ensure RequiredParameters imported from deprecated module works correctly."""
    # pylint: disable=import-outside-toplevel
    from firecrown.parameters import RequiredParameters, SamplerParameter

    req_params = RequiredParameters(
        [
            SamplerParameter(name="param1", default_value=1.0),
            SamplerParameter(name="param2", default_value=2.0),
        ]
    )
    assert "param1" in req_params.get_params_names()
    assert "param2" in req_params.get_params_names()


def test_derived_parameter_from_deprecated_module():
    """Ensure DerivedParameter imported from deprecated module works correctly."""
    # pylint: disable=import-outside-toplevel
    from firecrown.parameters import DerivedParameter

    derived = DerivedParameter("derived_parameters", "sum", 3.14)
    assert derived.name == "sum"
    assert derived.section == "derived_parameters"
    assert derived.val == 3.14
