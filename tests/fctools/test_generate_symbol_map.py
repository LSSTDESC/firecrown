"""Tests for the generate_symbol_map module.

Test imports must be inside functions to avoid circular dependencies
and to test dynamic importing behavior.
"""

import json
from types import ModuleType

import pytest
from typer.testing import CliRunner

from firecrown.fctools.generate_symbol_map import (
    _is_api_constant,
    _is_private_name,
    _is_excluded_type,
    _has_constant_name,
    _is_firecrown_instance,
    _add_symbol_to_map,
    get_all_symbols,
    app,
)


# Test fixtures - create some mock objects


class MockFirecrownClass:
    """Mock class that appears to be from firecrown package."""

    __module__ = "firecrown.mock.module"


class MockExternalClass:
    """Mock class that appears to be from external package."""

    __module__ = "external.package"


def mock_function():
    """Mock function for testing."""


mock_function.__module__ = "firecrown.mock.module"


def test_is_private_name():
    """Test that private names are detected."""
    assert _is_private_name("_private")
    assert _is_private_name("__dunder__")
    assert not _is_private_name("public")
    assert not _is_private_name("MY_CONSTANT")


def test_is_excluded_type_class():
    """Test that classes are identified as excluded types."""
    assert _is_excluded_type(MockFirecrownClass)


def test_is_excluded_type_function():
    """Test that functions are identified as excluded types."""
    assert _is_excluded_type(mock_function)


def test_is_excluded_type_module():
    """Test that modules are identified as excluded types."""
    mock_module = ModuleType("mock")
    assert _is_excluded_type(mock_module)


def test_is_excluded_type_instance():
    """Test that instances are not excluded types."""
    instance = MockFirecrownClass()
    assert not _is_excluded_type(instance)
    assert not _is_excluded_type(42)
    assert not _is_excluded_type("value")


def test_has_constant_name_uppercase():
    """Test that UPPER_CASE names are recognized."""
    assert _has_constant_name("CONSTANT")
    assert _has_constant_name("MY_CONSTANT")
    assert _has_constant_name("API_KEY")
    assert _has_constant_name("Y1_LENS_BINS")
    assert _has_constant_name("FOUND4HELP")


def test_has_constant_name_mixed_case():
    """Test that Mixed_Case names are NOT recognized.

    The regex pattern requires all-uppercase constant names.
    """
    assert not _has_constant_name("My_Constant")
    assert not _has_constant_name("Some_Constant")


def test_has_constant_name_lowercase():
    """Test that lowercase names are not recognized."""
    assert not _has_constant_name("lowercase")
    assert not _has_constant_name("my_var")
    assert not _has_constant_name("some_variable")


def test_is_firecrown_instance_with_firecrown_class():
    """Test that instances of Firecrown classes are identified."""
    instance = MockFirecrownClass()
    assert _is_firecrown_instance(instance)


def test_is_firecrown_instance_with_external_class():
    """Test that instances of external classes are not identified."""
    instance = MockExternalClass()
    assert not _is_firecrown_instance(instance)


def test_is_firecrown_instance_with_builtin():
    """Test that builtin instances are not identified as Firecrown."""
    assert not _is_firecrown_instance(42)
    assert not _is_firecrown_instance("value")
    assert not _is_firecrown_instance([1, 2, 3])
    assert not _is_firecrown_instance({"key": "value"})


def test_is_api_constant_private_name():
    """Test that private names are rejected."""
    assert not _is_api_constant("_private", "anything")
    assert not _is_api_constant("__dunder__", "anything")


def test_is_api_constant_class():
    """Test that classes are rejected (handled separately)."""
    assert not _is_api_constant("SomeClass", MockFirecrownClass)


def test_is_api_constant_function():
    """Test that functions are rejected (handled separately)."""
    assert not _is_api_constant("some_function", mock_function)


def test_is_api_constant_module():
    """Test that modules are rejected."""
    mock_module = ModuleType("mock")
    assert not _is_api_constant("some_module", mock_module)


def test_is_api_constant_uppercase():
    """Test that UPPER_CASE names are accepted."""
    assert _is_api_constant("CONSTANT", 42)
    assert _is_api_constant("MY_CONSTANT", "value")
    assert _is_api_constant("API_KEY", [1, 2, 3])


def test_is_api_constant_mixed_case_with_underscore():
    """Test that Mixed_Case_With_Underscores names are NOT accepted.

    The regex pattern requires all-uppercase constant names.
    Mixed case names don't follow Python constant naming conventions.
    """
    assert not _is_api_constant("My_Constant", {"key": "value"})
    # But all-uppercase with underscores IS accepted
    assert _is_api_constant("MY_CONSTANT", {"key": "value"})
    assert _is_api_constant("Y1_LENS_BINS", {"bin1": 0.1})


def test_is_api_constant_lowercase():
    """Test that lowercase names are rejected."""
    assert not _is_api_constant("lowercase", "value")
    assert not _is_api_constant("my_var", 123)


def test_is_api_constant_firecrown_instance():
    """Test that instances of Firecrown classes are accepted."""
    instance = MockFirecrownClass()
    assert _is_api_constant("SomeInstance", instance)


def test_is_api_constant_external_instance():
    """Test that instances of external classes are rejected."""
    instance = MockExternalClass()
    assert not _is_api_constant("ExternalInstance", instance)


def test_add_symbol_to_map_class():
    """Test adding a class to the symbol map."""
    symbols: dict[str, str] = {}
    _add_symbol_to_map(
        symbols, "test.module", "TestClass", MockFirecrownClass, "firecrown"
    )

    assert "test.module.TestClass" in symbols
    assert (
        symbols["test.module.TestClass"]
        == "api/firecrown.mock.module.html#firecrown.mock.module.TestClass"
    )


def test_add_symbol_to_map_function():
    """Test adding a function to the symbol map."""
    symbols: dict[str, str] = {}
    _add_symbol_to_map(symbols, "test.module", "test_func", mock_function, "firecrown")

    assert "test.module.test_func" in symbols
    assert (
        symbols["test.module.test_func"]
        == "api/firecrown.mock.module.html#firecrown.mock.module.test_func"
    )


def test_add_symbol_to_map_re_exported_class():
    """Test that re-exported classes get both public and private paths."""
    symbols: dict[str, str] = {}

    # Simulate a class defined in _private but exported through public
    _add_symbol_to_map(
        symbols, "firecrown.public", "MyClass", MockFirecrownClass, "firecrown"
    )

    # Should have public path
    assert "firecrown.public.MyClass" in symbols

    # Should also have defining module path (since they differ)
    assert "firecrown.mock.module.MyClass" in symbols


def test_add_symbol_to_map_external_class():
    """Test that external classes are not added."""
    symbols: dict[str, str] = {}
    _add_symbol_to_map(
        symbols, "test.module", "ExternalClass", MockExternalClass, "firecrown"
    )

    assert len(symbols) == 0


def test_add_symbol_to_map_constant():
    """Test adding a constant to the symbol map."""
    symbols: dict[str, str] = {}
    my_constant = {"key": "value"}

    _add_symbol_to_map(symbols, "test.module", "MY_CONSTANT", my_constant, "firecrown")

    assert "test.module.MY_CONSTANT" in symbols
    expected_url = "api/test.module.html#test.module.MY_CONSTANT"
    assert symbols["test.module.MY_CONSTANT"] == expected_url


def test_add_symbol_to_map_mixed_case_constant():
    """Test adding a Mixed_Case constant to the symbol map."""
    symbols: dict[str, str] = {}
    bins = {"bin1": 0.1, "bin2": 0.2}

    _add_symbol_to_map(symbols, "test.module", "Y1_LENS_BINS", bins, "firecrown")

    assert "test.module.Y1_LENS_BINS" in symbols


def test_add_symbol_to_map_firecrown_instance():
    """Test adding a Firecrown class instance to the symbol map."""
    symbols: dict[str, str] = {}
    instance = MockFirecrownClass()

    _add_symbol_to_map(symbols, "test.module", "INSTANCE", instance, "firecrown")

    assert "test.module.INSTANCE" in symbols


def test_add_symbol_to_map_lowercase_variable():
    """Test that lowercase variables are not added."""
    symbols: dict[str, str] = {}
    _add_symbol_to_map(symbols, "test.module", "lowercase_var", 42, "firecrown")

    assert len(symbols) == 0


def test_add_symbol_to_map_private_constant():
    """Test that private constants are not added."""
    symbols: dict[str, str] = {}
    _add_symbol_to_map(symbols, "test.module", "_PRIVATE_CONST", 42, "firecrown")

    assert len(symbols) == 0


def test_get_all_symbols_with_mock_package():
    """Test get_all_symbols with a mock package structure."""
    # Create a minimal mock package
    mock_package = ModuleType("mock_package")
    mock_package.__path__ = []  # Empty path to avoid walking real modules

    # Mock pkgutil.walk_packages to return no submodules
    import pkgutil  # pylint: disable=import-outside-toplevel

    original_walk = pkgutil.walk_packages

    def mock_walk(*_args, **_kwargs):
        return []

    pkgutil.walk_packages = mock_walk

    try:
        symbols = get_all_symbols(mock_package)
        # Should at least handle the empty case without crashing
        assert isinstance(symbols, dict)
    finally:
        pkgutil.walk_packages = original_walk


def test_get_all_symbols_real_firecrown_subset():
    """Test get_all_symbols with real firecrown.updatable module."""
    import firecrown.updatable  # pylint: disable=import-outside-toplevel

    symbols = get_all_symbols(firecrown.updatable)

    # Should have captured some symbols
    assert len(symbols) > 0

    # Should have some classes from updatable
    # (without hard-coding specific class names that might change)
    param_symbols = [k for k in symbols if "firecrown.updatable." in k]
    assert len(param_symbols) > 0


def test_get_all_symbols_captures_classes():
    """Test that get_all_symbols captures class definitions."""
    import firecrown.updatable  # pylint: disable=import-outside-toplevel

    symbols = get_all_symbols(firecrown.updatable)

    # Look for any class-like symbols (conventionally CamelCase)
    classes = [
        k
        for k in symbols
        if k.split(".")[-1][0].isupper() and "_" not in k.split(".")[-1]
    ]
    assert len(classes) > 0


def test_get_all_symbols_captures_functions():
    """Test that get_all_symbols captures function definitions."""
    import firecrown.updatable  # pylint: disable=import-outside-toplevel

    symbols = get_all_symbols(firecrown.updatable)

    # Should have some symbols (classes or functions)
    assert len(symbols) > 0


def test_get_all_symbols_excludes_private():
    """Test that get_all_symbols excludes private members.

    The function excludes members with names starting with underscore,
    but includes private module names (like _parameters_derived).
    """
    import firecrown.updatable  # pylint: disable=import-outside-toplevel

    symbols = get_all_symbols(firecrown.updatable)

    # Should have included private module names like _parameters_derived
    assert any("._parameters_derived" in k for k in symbols)

    # Check a real example: _parameters_derived module exists, but not _private_function
    # All symbols from _parameters_derived should be public classes/functions
    derived_symbols = [k for k in symbols if "._parameters_derived." in k]
    for sym in derived_symbols:
        # The part after _parameters_derived. should not start with underscore
        after_derived = sym.split("._parameters_derived.")[-1]
        assert not after_derived.startswith("_"), f"Found private member: {sym}"


def test_get_all_symbols_with_generators():
    """Test get_all_symbols captures constants from generators."""
    import firecrown.generators  # pylint: disable=import-outside-toplevel

    symbols = get_all_symbols(firecrown.generators)

    # Should capture UPPER_CASE constants
    const_symbols = [
        k
        for k in symbols
        if k.split(".")[-1].isupper()
        or ("_" in k.split(".")[-1] and k.split(".")[-1][0].isupper())
    ]

    # Should have captured some constants
    assert len(const_symbols) > 0

    # Specifically check for known constants if they exist
    if any("Y1_LENS_BINS" in k for k in symbols):
        assert any("firecrown.generators.Y1_LENS_BINS" in k for k in symbols)


# CLI Tests using CliRunner
runner = CliRunner()


@pytest.mark.filterwarnings(
    r"ignore:.*firecrown\.ccl_factory is deprecated.*:DeprecationWarning"
)
@pytest.mark.filterwarnings(
    r"ignore:.*firecrown\.parameters module is deprecated.*:DeprecationWarning"
)
def test_main_outputs_valid_json():
    """Test that main command outputs valid JSON to stdout."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0

    # Parse the JSON output
    data = json.loads(result.stdout)

    # Verify structure
    assert isinstance(data, dict)
    assert len(data) > 0

    # Check some expected symbols exist
    assert any("firecrown" in key for key in data)


def test_main_with_output_file(tmp_path):
    """Test main command with --output option writes to file."""
    output_file = tmp_path / "symbols.json"

    result = runner.invoke(app, ["--output", str(output_file)])
    assert result.exit_code == 0

    # Verify file was created
    assert output_file.exists()

    # Verify file contains valid JSON
    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0


def test_main_with_compact_format(tmp_path):
    """Test --compact flag produces single-line JSON."""
    output_file = tmp_path / "compact.json"

    result = runner.invoke(app, ["--output", str(output_file), "--compact"])
    assert result.exit_code == 0

    # Read raw file content
    content = output_file.read_text()

    # Compact JSON should be on a single line (0 or 1 newlines total)
    assert content.count("\n") <= 1

    # Verify it's valid JSON
    data = json.loads(content)
    assert len(data) > 0


def test_main_default_is_pretty(tmp_path):
    """Test that default formatting is pretty (multi-line)."""
    output_file = tmp_path / "pretty.json"

    result = runner.invoke(app, ["--output", str(output_file)])
    assert result.exit_code == 0

    # Read raw file content
    content = output_file.read_text()

    # Pretty JSON should have multiple newlines
    assert content.count("\n") > 10


def test_main_symbol_count():
    """Test that main command reports symbol count."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0

    # Parse the JSON to count symbols
    data = json.loads(result.stdout)
    symbol_count = len(data)

    # Should have found a reasonable number of symbols
    assert symbol_count > 50


def test_main_output_with_status_message(tmp_path):
    """Test that main command shows status message when writing to file."""
    output_file = tmp_path / "symbols.json"

    result = runner.invoke(app, ["--output", str(output_file)])
    assert result.exit_code == 0

    # Verify status message appears in stderr
    assert "Generated" in result.stderr
    assert "symbols" in result.stderr


def test_is_api_constant_with_non_firecrown_class():
    """Test _is_api_constant with object that has class with non-firecrown module."""
    # Create an object with a class that has __module__ but not from firecrown
    obj = object()  # object's __class__.__module__ is 'builtins'

    result = _is_api_constant("test_obj", obj)

    # Should return False since the class module doesn't start with firecrown
    assert result is False


def test_is_api_constant_mixed_case_only_second_condition():
    """Test _is_api_constant with various naming patterns.

    The regex pattern now requires proper UPPER_CASE constant naming.
    Mixed case names are rejected.
    """

    class PretendConstant:
        """Mock class for testing constant naming patterns."""

    # Mixed case with underscore is NOT accepted
    result = _is_api_constant("Some_Constant", PretendConstant())
    assert result is False

    # All uppercase with underscore IS accepted
    result = _is_api_constant("SOME_CONSTANT", PretendConstant())
    assert result is True

    # Lowercase with underscore is NOT accepted
    result = _is_api_constant("some_variable", PretendConstant())
    assert result is False
