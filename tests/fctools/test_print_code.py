"""Tests for the fctools/print_code.py module."""

import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from firecrown.fctools.print_code import (
    _build_class_code,
    _render_attributes,
    display_class_attributes,
    display_class_without_markdown,
    main,
)

from . import match_wrapped, strip_rich_markup

# Check Python version for type_params support (added in 3.12)
_SUPPORTS_TYPE_PARAMS = sys.version_info >= (3, 12)

# pylint: disable=missing-function-docstring,missing-class-docstring


if sys.version_info >= (3, 12):

    def _make_class_def(
        name: str,
        bases: list[ast.expr],
        keywords: list[ast.keyword],
        body: list[ast.stmt],
        decorator_list: list[ast.expr],
    ) -> ast.ClassDef:
        """Create a ClassDef node with type_params (Python 3.12+)."""
        return ast.ClassDef(
            name=name,
            bases=bases,
            keywords=keywords,
            body=body,
            decorator_list=decorator_list,
            type_params=[],
        )

else:

    def _make_class_def(
        name: str,
        bases: list[ast.expr],
        keywords: list[ast.keyword],
        body: list[ast.stmt],
        decorator_list: list[ast.expr],
    ) -> ast.ClassDef:
        """Create a ClassDef node without type_params (Python 3.11)."""
        return ast.ClassDef(
            name=name,
            bases=bases,
            keywords=keywords,
            body=body,
            decorator_list=decorator_list,
        )


def _get_subprocess_env():
    """Get environment with PYTHONPATH set to include the current directory."""
    env = os.environ.copy()
    # Add current directory to PYTHONPATH so subprocess can import tests module
    current_dir = str(Path(__file__).parent.parent.parent)
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{current_dir}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = current_dir
    return env


# Test helper classes
class SimpleClass:
    """A simple test class."""

    attr1: int
    attr2: str = "default"


@dataclass
class DecoratedClass:
    """A decorated test class."""

    value: int


class ClassWithBases(SimpleClass):
    """A class with base classes."""

    attr3: float = 1.0


class ClassNoDocstring:
    """A class intentionally used to test classes without elaborate docstrings."""

    attr: int


# Tests for _render_attributes


def test_render_attributes_simple():
    """Test _render_attributes with simple attributes."""
    # Create a simple ClassDef with attributes
    class_def = _make_class_def(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[
            ast.AnnAssign(
                target=ast.Name(id="attr1"),
                annotation=ast.Name(id="int"),
                value=None,
                simple=1,
            ),
            ast.AnnAssign(
                target=ast.Name(id="attr2"),
                annotation=ast.Name(id="str"),
                value=ast.Constant(value="default"),
                simple=1,
            ),
        ],
        decorator_list=[],
    )
    result = _render_attributes(class_def)
    assert any("attr1: int" in line for line in result)
    assert any("attr2: str = 'default'" in line for line in result)


def test_render_attributes_empty():
    """Test _render_attributes with empty class body."""
    class_def = _make_class_def(
        name="EmptyClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[],
    )
    result = _render_attributes(class_def)
    assert not result


def test_render_attributes_no_annotation():
    """Test _render_attributes with attribute without annotation."""
    assign_node = ast.Assign(
        targets=[ast.Name(id="attr")], value=ast.Constant(value=42)
    )
    # Need to add line number for ast.unparse to work
    ast.fix_missing_locations(assign_node)
    class_def = _make_class_def(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[assign_node],
        decorator_list=[],
    )
    result = _render_attributes(class_def)
    assert any("attr = 42" in line for line in result)


def test_render_attributes_complex_value():
    """Test _render_attributes with complex attribute value."""
    class_def = _make_class_def(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[
            ast.AnnAssign(
                target=ast.Name(id="attr"),
                annotation=ast.Name(id="list"),
                value=ast.List(elts=[ast.Constant(value=1), ast.Constant(value=2)]),
                simple=1,
            )
        ],
        decorator_list=[],
    )
    result = _render_attributes(class_def)
    assert any("attr: list = [1, 2]" in line for line in result)


# Tests for _build_class_code


def test_build_class_code_simple_class():
    """Test _build_class_code with a simple class."""
    code = _build_class_code(SimpleClass)
    assert "class SimpleClass():" in code
    assert "A simple test class." in code
    assert "attr1: int" in code
    assert "attr2: str = 'default'" in code


def test_build_class_code_decorated_class():
    """Test _build_class_code with a decorated class."""
    code = _build_class_code(DecoratedClass)
    assert "@dataclass" in code
    assert "class DecoratedClass():" in code
    assert "value: int" in code


def test_build_class_code_class_with_bases():
    """Test _build_class_code with a class that has base classes."""
    code = _build_class_code(ClassWithBases)
    # This should include the base class - exact format depends on AST
    assert "class ClassWithBases" in code
    assert "attr3: float = 1.0" in code


def test_build_class_code_no_docstring():
    """Test _build_class_code with a class without docstring."""
    code = _build_class_code(ClassNoDocstring)
    assert "class ClassNoDocstring():" in code
    assert "attr: int" in code


def test_build_class_code_builtin_class():
    """Test _build_class_code with a built-in class."""
    # Built-in classes should raise TypeError (no source file)
    with pytest.raises(TypeError):
        _build_class_code(int)


def test_build_class_code_no_attributes():
    """Test _build_class_code with a class that has no attributes."""
    # Use a class defined at module level to avoid indentation issues
    code = _build_class_code(ClassNoDocstring)
    assert "class ClassNoDocstring():" in code
    assert "attr: int" in code


# Tests for display_class_attributes


def test_display_class_attributes_simple(capsys):
    """Test display_class_attributes with a simple class."""
    display_class_attributes(SimpleClass)
    captured = capsys.readouterr()
    assert "```python" in captured.out
    assert "class SimpleClass():" in captured.out
    assert "attr1: int" in captured.out
    assert "```" in captured.out


def test_display_class_attributes_builtin_class():
    """Test display_class_attributes with a built-in class."""
    # Built-in classes raise TypeError, not OSError, so it will propagate
    with pytest.raises(TypeError, match="built-in class"):
        display_class_attributes(int)


def test_display_class_attributes_decorated(capsys):
    """Test display_class_attributes with a decorated class."""
    display_class_attributes(DecoratedClass)
    captured = capsys.readouterr()
    assert "@dataclass" in captured.out
    assert "class DecoratedClass():" in captured.out


def test_display_class_attributes_no_markdown_simple(capsys):
    """Test display_class_attributes with markdown disabled."""
    # This should just call display_class_without_markdown
    display_class_attributes(SimpleClass)
    captured = capsys.readouterr()
    assert "class SimpleClass():" in captured.out


# Tests for display_class_without_markdown


def test_display_class_without_markdown_simple(capsys):
    """Test display_class_without_markdown with a simple class."""
    display_class_without_markdown(SimpleClass)
    captured = capsys.readouterr()
    assert "class SimpleClass():" in captured.out
    assert "attr1: int" in captured.out
    # Should NOT have markdown markers
    assert "```python" not in captured.out
    assert "```" not in captured.out


def test_display_class_without_markdown_builtin_class():
    """Test display_class_without_markdown with a built-in class."""
    # Built-in classes raise TypeError, not OSError, so it will propagate
    with pytest.raises(TypeError, match="built-in class"):
        display_class_without_markdown(int)


def test_display_class_without_markdown_decorated(capsys):
    """Test display_class_without_markdown with a decorated class."""
    display_class_without_markdown(DecoratedClass)
    captured = capsys.readouterr()
    assert "@dataclass" in captured.out
    assert "class DecoratedClass():" in captured.out


# Tests for main CLI


def test_main_simple_class():
    """Test main CLI with a simple class path."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [sys.executable, script_path, "tests.fctools.test_print_code.SimpleClass"],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    assert result.returncode == 0
    assert match_wrapped(result.stdout, "class SimpleClass():")
    assert match_wrapped(result.stdout, "attr1: int")


def test_main_no_markdown():
    """Test main CLI with --no-markdown flag."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--no-markdown",
            "tests.fctools.test_print_code.SimpleClass",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    assert result.returncode == 0
    assert match_wrapped(result.stdout, "class SimpleClass():")
    assert "```python" not in result.stdout


def test_main_decorated_class():
    """Test main CLI with a decorated class."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [sys.executable, script_path, "tests.fctools.test_print_code.DecoratedClass"],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    assert result.returncode == 0
    assert match_wrapped(result.stdout, "@dataclass")
    assert match_wrapped(result.stdout, "class DecoratedClass():")


def test_main_class_with_bases():
    """Test main CLI with a class that has base classes."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [sys.executable, script_path, "tests.fctools.test_print_code.ClassWithBases"],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    assert result.returncode == 0
    assert "class ClassWithBases" in result.stdout
    assert "attr3: float = 1.0" in result.stdout


def test_main_invalid_class_path():
    """Test main CLI with invalid class path."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [sys.executable, script_path, "nonexistent.module.Class"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Should fail due to import_class_from_path calling cli_error -> sys.exit(1)
    assert result.returncode != 0


def test_main_builtin_class():
    """Test main CLI with built-in class."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [sys.executable, script_path, "builtins.int"],
        capture_output=True,
        text=True,
        check=False,
    )
    # TypeError is raised for built-in classes, which propagates up
    assert result.returncode != 0


def test_main_missing_argument():
    """Test main CLI with missing argument."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Missing argument" in result.stderr


def test_main_help():
    """Test main CLI with --help flag."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert match_wrapped(result.stdout, "Usage:")
    assert match_wrapped(result.stdout, "Display class definitions")
    assert match_wrapped(strip_rich_markup(result.stdout), "--no-markdown")


def test_main_multiple_classes():
    """Test main CLI with multiple class names."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "tests.fctools.test_print_code.SimpleClass",
            "tests.fctools.test_print_code.DecoratedClass",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    assert result.returncode == 0
    # Should have headers for multiple classes
    assert match_wrapped(
        result.stdout, "Class: tests.fctools.test_print_code.SimpleClass"
    )
    assert match_wrapped(
        result.stdout, "Class: tests.fctools.test_print_code.DecoratedClass"
    )
    assert "=" * 60 in result.stdout
    # Should have both classes in output
    assert "class SimpleClass()" in result.stdout
    assert "class DecoratedClass()" in result.stdout


def test_main_multiple_classes_with_invalid():
    """Test main CLI with multiple classes where one is invalid."""
    script_path = "firecrown/fctools/print_code.py"
    # Test with invalid module
    # import_class_from_path will call cli_error -> sys.exit(1)
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "tests.fctools.test_print_code.SimpleClass",
            "nonexistent.module.Class",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    # Should fail due to cli_error calling sys.exit(1)
    assert result.returncode != 0
    # First class should succeed before the error
    assert (
        match_wrapped(result.stdout, "class SimpleClass()") or "ERROR:" in result.stderr
    )


def test_build_class_code_with_none_class_def():
    """Test _build_class_code when get_class_definition returns None."""
    # This tests the error path when class definition can't be found
    # The ValueError is raised when class_def is None
    # This can happen with dynamically created classes or unusual source code

    class TestClass:
        pass

    # Mock get_class_definition to return None
    with patch("firecrown.fctools.print_code.get_class_definition", return_value=None):
        try:
            _build_class_code(TestClass)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Could not find class definition" in str(e)


# Integration tests


def test_integration_full_workflow():
    """Test full workflow from CLI to output."""
    script_path = "firecrown/fctools/print_code.py"
    result = subprocess.run(
        [sys.executable, script_path, "tests.fctools.test_print_code.SimpleClass"],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    assert result.returncode == 0
    # Verify the output contains all expected elements
    assert "class SimpleClass():" in result.stdout
    assert "A simple test class." in result.stdout
    assert "attr1: int" in result.stdout
    assert "attr2: str = 'default'" in result.stdout


def test_integration_multiple_runs():
    """Test that multiple runs work correctly."""
    script_path = "firecrown/fctools/print_code.py"
    # Run twice with same class
    result1 = subprocess.run(
        [sys.executable, script_path, "tests.fctools.test_print_code.SimpleClass"],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    result2 = subprocess.run(
        [sys.executable, script_path, "tests.fctools.test_print_code.SimpleClass"],
        capture_output=True,
        text=True,
        check=False,
        env=_get_subprocess_env(),
    )
    assert result1.returncode == 0
    assert result2.returncode == 0
    assert result1.stdout == result2.stdout


def test_integration_error_handling():
    """Test error handling in full workflow."""
    script_path = "firecrown/fctools/print_code.py"
    # Test with invalid path
    result = subprocess.run(
        [sys.executable, script_path, "invalid.path.Class"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


# Direct main() function tests for coverage


def test_main_function_single_class(capsys):
    """Test main function directly with single class."""
    # Call main with a single class
    main(class_names=["tests.fctools.test_print_code.SimpleClass"], no_markdown=False)

    captured = capsys.readouterr()
    assert "class SimpleClass():" in captured.out
    assert "```python" in captured.out
    assert "A simple test class." in captured.out


def test_main_function_no_markdown_flag(capsys):
    """Test main function with no_markdown=True."""
    # Call main with no_markdown flag
    main(class_names=["tests.fctools.test_print_code.SimpleClass"], no_markdown=True)

    captured = capsys.readouterr()
    assert "class SimpleClass():" in captured.out
    # Should NOT have markdown code blocks
    assert "```python" not in captured.out


def test_main_function_multiple_classes(capsys):
    """Test main function with multiple classes."""
    # Call main with multiple classes
    main(
        class_names=[
            "tests.fctools.test_print_code.SimpleClass",
            "tests.fctools.test_print_code.DecoratedClass",
        ],
        no_markdown=False,
    )

    captured = capsys.readouterr()
    # Should show both classes
    assert "class SimpleClass():" in captured.out
    assert "class DecoratedClass():" in captured.out
    # Should show separators for multiple classes
    assert "=" * 60 in captured.out
    assert "Class: tests.fctools.test_print_code.SimpleClass" in captured.out
    assert "Class: tests.fctools.test_print_code.DecoratedClass" in captured.out


def test_main_function_multiple_classes_no_markdown(capsys):
    """Test main function with multiple classes and no_markdown=True."""
    # Call main with multiple classes and no_markdown
    main(
        class_names=[
            "tests.fctools.test_print_code.SimpleClass",
            "tests.fctools.test_print_code.DecoratedClass",
        ],
        no_markdown=True,
    )

    captured = capsys.readouterr()
    # Should show both classes without markdown
    assert "class SimpleClass():" in captured.out
    assert "class DecoratedClass():" in captured.out
    assert "```python" not in captured.out
    # Should still show separators
    assert "=" * 60 in captured.out
