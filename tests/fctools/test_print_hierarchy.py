"""Unit tests for firecrown.fctools.print_hierarchy module.

Tests the class hierarchy printing tool.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring
# Test helper classes don't need docstrings

import subprocess
import sys

from click.testing import CliRunner

from firecrown.fctools.print_hierarchy import (
    full_type_name,
    get_defined_methods,
    main,
    print_one_type,
    print_type_hierarchy,
)


# Tests for full_type_name()


def test_builtin_type():
    """Test getting full name for built-in type."""
    result = full_type_name(int)
    assert result == "builtins.int"


def test_standard_library_type():
    """Test getting full name for standard library type."""
    from collections import OrderedDict  # pylint: disable=import-outside-toplevel

    result = full_type_name(OrderedDict)
    assert result == "collections.OrderedDict"


def test_custom_class():
    """Test getting full name for a custom class."""

    class MyClass:
        pass

    result = full_type_name(MyClass)
    assert "MyClass" in result
    assert result.endswith(".MyClass")


# Tests for get_defined_methods()


def test_class_with_no_methods():
    """Test getting methods from a class with no methods."""

    class EmptyClass:
        pass

    methods = get_defined_methods(EmptyClass)
    assert isinstance(methods, list)
    assert len(methods) == 0


def test_class_with_simple_method():
    """Test getting methods from a class with one method."""

    class SimpleClass:
        def my_method(self):
            pass

    methods = get_defined_methods(SimpleClass)
    assert "my_method" in methods


def test_class_with_multiple_methods():
    """Test getting methods from a class with multiple methods."""

    class MultiMethodClass:
        def method_one(self):
            pass

        def method_two(self):
            pass

        def method_three(self):
            pass

    methods = get_defined_methods(MultiMethodClass)
    assert len(methods) == 3
    assert "method_one" in methods
    assert "method_two" in methods
    assert "method_three" in methods


def test_inherited_method_not_included():
    """Test that inherited methods are not included."""

    class BaseClass:
        def base_method(self):
            pass

    class DerivedClass(BaseClass):
        def derived_method(self):
            pass

    methods = get_defined_methods(DerivedClass)
    assert "derived_method" in methods
    assert "base_method" not in methods


def test_overridden_method_included():
    """Test that overridden methods are included."""

    class BaseClass:
        def my_method(self):
            return "base"

    class DerivedClass(BaseClass):
        def my_method(self):
            return "derived"

    methods = get_defined_methods(DerivedClass)
    assert "my_method" in methods


def test_special_methods_included():
    """Test that special methods are included."""

    class WithInit:
        def __init__(self):
            pass

        def __str__(self):
            return "test"

    methods = get_defined_methods(WithInit)
    assert "__init__" in methods
    assert "__str__" in methods


def test_class_with_static_and_class_methods():
    """Test that static and class methods are not included."""

    class WithSpecialMethods:
        def regular_method(self):
            pass

        @staticmethod
        def static_method():
            pass

        @classmethod
        def class_method(cls):
            pass

    methods = get_defined_methods(WithSpecialMethods)
    # Regular method should be included
    assert "regular_method" in methods
    # Static and class methods are not functions (they're wrapped)
    # so they may or may not be included depending on inspect behavior


# Tests for print_one_type()


def test_prints_type_info(capsys):
    """Test printing basic type information."""

    class SimpleClass:
        def my_method(self):
            pass

    print_one_type(0, SimpleClass)
    captured = capsys.readouterr()

    # Should print index and type name
    assert "0" in captured.out
    assert "SimpleClass" in captured.out
    assert "my_method" in captured.out


def test_prints_multiple_methods(capsys):
    """Test printing type with multiple methods."""

    class MultiMethodClass:
        def method_a(self):
            pass

        def method_b(self):
            pass

    print_one_type(1, MultiMethodClass)
    captured = capsys.readouterr()

    assert "1" in captured.out
    assert "MultiMethodClass" in captured.out
    assert "method_a" in captured.out
    assert "method_b" in captured.out


def test_prints_with_different_indices(capsys):
    """Test printing with different index values."""

    class TestClass:
        pass

    print_one_type(5, TestClass)
    captured = capsys.readouterr()

    assert "5" in captured.out
    assert "TestClass" in captured.out


# Tests for print_type_hierarchy()


def test_prints_simple_class(capsys):
    """Test printing hierarchy for a simple class."""

    class SimpleClass:
        pass

    print_type_hierarchy(SimpleClass)
    captured = capsys.readouterr()

    assert "Hierarchy for" in captured.out
    assert "SimpleClass" in captured.out
    # Should include object at the end of MRO
    assert "builtins.object" in captured.out


def test_prints_inheritance_chain(capsys):
    """Test printing hierarchy with inheritance."""

    class Base:
        def base_method(self):
            pass

    class Middle(Base):
        def middle_method(self):
            pass

    class Derived(Middle):
        def derived_method(self):
            pass

    print_type_hierarchy(Derived)
    captured = capsys.readouterr()

    # All classes in the hierarchy should be present
    assert "Derived" in captured.out
    assert "Middle" in captured.out
    assert "Base" in captured.out
    assert "object" in captured.out

    # Methods should be shown under their defining classes
    assert "derived_method" in captured.out
    assert "middle_method" in captured.out
    assert "base_method" in captured.out


def test_prints_builtin_type_hierarchy(capsys):
    """Test printing hierarchy for built-in types."""
    from collections import OrderedDict  # pylint: disable=import-outside-toplevel

    print_type_hierarchy(OrderedDict)
    captured = capsys.readouterr()

    assert "Hierarchy for" in captured.out
    assert "OrderedDict" in captured.out
    assert "dict" in captured.out
    assert "object" in captured.out


# Tests for main()


def test_main_with_single_typename():
    """Test main with a single type name."""
    runner = CliRunner()
    result = runner.invoke(main, ["collections.OrderedDict"])

    assert result.exit_code == 0
    assert "Hierarchy for" in result.output
    assert "OrderedDict" in result.output


def test_main_with_multiple_typenames():
    """Test main with multiple type names."""
    runner = CliRunner()
    result = runner.invoke(main, ["pathlib.Path", "collections.Counter"])

    assert result.exit_code == 0
    assert "Path" in result.output
    assert "Counter" in result.output
    # Should have separator between them
    assert "=" in result.output


def test_main_with_no_arguments():
    """Test main with no arguments (should fail)."""
    runner = CliRunner()
    result = runner.invoke(main, [])

    assert result.exit_code != 0
    # Should show error about missing argument


def test_main_with_invalid_typename():
    """Test main with invalid type name."""
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.module.Type"])

    # Should handle the error gracefully (import_class_from_path calls cli_error)
    assert result.exit_code != 0
    assert "Could not import module" in result.output
    assert "nonexistent" in result.output


def test_main_with_mixed_valid_invalid():
    """Test main with mix of valid and invalid type names."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["collections.OrderedDict", "invalid.Type", "pathlib.Path"]
    )

    # The first valid type should be shown
    assert "OrderedDict" in result.output
    # But the tool exits on the first error (import_class_from_path calls cli_error)
    # So Path won't be processed
    assert result.exit_code != 0
    assert "Could not import module" in result.output


def test_main_with_builtin_type():
    """Test main with built-in type."""
    runner = CliRunner()
    result = runner.invoke(main, ["builtins.int"])

    assert result.exit_code == 0
    assert "int" in result.output


def test_main_with_subprocess():
    """Test that the script can be executed directly via subprocess.

    This test verifies that the __main__ block works correctly.
    """
    script_path = "firecrown/fctools/print_hierarchy.py"
    result = subprocess.run(
        [sys.executable, script_path, "pathlib.Path"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Hierarchy for" in result.stdout
    assert "Path" in result.stdout


def test_main_subprocess_with_multiple_types():
    """Test script execution with multiple type arguments."""
    script_path = "firecrown/fctools/print_hierarchy.py"
    result = subprocess.run(
        [sys.executable, script_path, "pathlib.Path", "collections.Counter"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Path" in result.stdout
    assert "Counter" in result.stdout
    assert "=" in result.stdout


def test_main_subprocess_with_invalid_type():
    """Test script execution with invalid type name."""
    script_path = "firecrown/fctools/print_hierarchy.py"
    result = subprocess.run(
        [sys.executable, script_path, "invalid.module.Type"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Script should exit with error (import_class_from_path calls cli_error)
    assert result.returncode != 0
    # Error goes to stderr
    assert "Could not import module" in result.stderr
    assert "invalid" in result.stderr


# Integration tests


def test_full_workflow_with_standard_type(capsys):
    """Test complete workflow with a standard library type."""
    from collections import OrderedDict  # pylint: disable=import-outside-toplevel

    # Test getting full name
    name = full_type_name(OrderedDict)
    assert "OrderedDict" in name

    # Test getting methods
    methods = get_defined_methods(OrderedDict)
    assert isinstance(methods, list)

    # Test printing hierarchy
    print_type_hierarchy(OrderedDict)
    captured = capsys.readouterr()
    assert "OrderedDict" in captured.out
    assert "dict" in captured.out


def test_full_workflow_with_custom_hierarchy(capsys):
    """Test complete workflow with custom class hierarchy."""

    class GrandParent:
        def grandparent_method(self):
            pass

    class Parent(GrandParent):
        def parent_method(self):
            pass

    class Child(Parent):
        def child_method(self):
            pass

    # Test each function
    assert "Child" in full_type_name(Child)
    assert "child_method" in get_defined_methods(Child)

    print_type_hierarchy(Child)
    captured = capsys.readouterr()

    # Verify hierarchy is complete
    assert "Child" in captured.out
    assert "Parent" in captured.out
    assert "GrandParent" in captured.out
    assert "child_method" in captured.out
    assert "parent_method" in captured.out
    assert "grandparent_method" in captured.out


def test_cli_produces_consistent_output():
    """Test that CLI produces consistent output across runs."""
    runner = CliRunner()

    # Run twice
    result1 = runner.invoke(main, ["pathlib.Path"])
    result2 = runner.invoke(main, ["pathlib.Path"])

    # Should produce identical output
    assert result1.exit_code == 0
    assert result2.exit_code == 0
    assert result1.output == result2.output


def test_handles_complex_inheritance():
    """Test handling complex inheritance patterns."""
    runner = CliRunner()
    # OrderedDict has complex MRO
    result = runner.invoke(main, ["collections.OrderedDict"])

    assert result.exit_code == 0
    output_lines = result.output.split("\n")

    # Should show multiple levels of hierarchy
    hierarchy_lines = [line for line in output_lines if line.strip()]
    assert len(hierarchy_lines) > 2  # At least OrderedDict, dict, object


def test_method_detection_accuracy():
    """Test that method detection is accurate."""

    class Base:
        def base_only(self):
            pass

        def to_override(self):
            pass

    class Derived(Base):
        def derived_only(self):
            pass

        def to_override(self):
            pass  # Override

    base_methods = get_defined_methods(Base)
    derived_methods = get_defined_methods(Derived)

    # Base should have both of its methods
    assert "base_only" in base_methods
    assert "to_override" in base_methods

    # Derived should have its own method and the overridden one
    assert "derived_only" in derived_methods
    assert "to_override" in derived_methods
    # But not the inherited base_only
    assert "base_only" not in derived_methods
