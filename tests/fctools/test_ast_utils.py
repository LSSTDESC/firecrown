"""Unit tests for firecrown.fctools.ast_utils module.

Tests the AST parsing and analysis utilities.
"""

import ast

import pytest

from firecrown.fctools.ast_utils import (
    extract_class_attributes,
    format_class_docstring,
    format_docstring_summary,
    get_class_definition,
    get_function_names,
    get_module_docstring,
)


class TestGetModuleDocstring:
    """Tests for get_module_docstring function."""

    def test_get_simple_docstring(self, tmp_path):
        """Test extracting a simple module docstring."""
        py_file = tmp_path / "test_module.py"
        py_file.write_text('"""This is a module docstring."""\n\ndef func():\n    pass')

        result = get_module_docstring(py_file)

        assert result == "This is a module docstring."

    def test_get_multiline_docstring(self, tmp_path):
        """Test extracting a multi-line module docstring."""
        py_file = tmp_path / "test_module.py"
        content = '''"""This is the first line.

This is the second paragraph.
"""

def func():
    pass
'''
        py_file.write_text(content)

        result = get_module_docstring(py_file)

        assert result is not None
        assert "This is the first line." in result
        assert "This is the second paragraph." in result

    def test_get_no_docstring(self, tmp_path):
        """Test file with no module docstring."""
        py_file = tmp_path / "test_module.py"
        py_file.write_text("def func():\n    pass")

        result = get_module_docstring(py_file)

        assert result is None

    def test_get_single_quote_docstring(self, tmp_path):
        """Test extracting docstring with single quotes."""
        py_file = tmp_path / "test_module.py"
        py_file.write_text("'''Single quote docstring'''\n\ndef func():\n    pass")

        result = get_module_docstring(py_file)

        assert result == "Single quote docstring"

    def test_nonexistent_file(self, tmp_path):
        """Test reading from a nonexistent file."""
        py_file = tmp_path / "nonexistent.py"

        with pytest.raises(OSError):
            get_module_docstring(py_file)

    def test_invalid_python_syntax(self, tmp_path):
        """Test file with invalid Python syntax."""
        py_file = tmp_path / "invalid.py"
        py_file.write_text("def broken(\n    # missing closing paren")

        with pytest.raises(SyntaxError):
            get_module_docstring(py_file)


class TestGetClassDefinition:
    """Tests for get_class_definition function."""

    def test_find_simple_class(self):
        """Test finding a simple class definition."""
        source = """
class MyClass:
    pass
"""
        result = get_class_definition(source, "MyClass")

        assert result is not None
        assert isinstance(result, ast.ClassDef)
        assert result.name == "MyClass"

    def test_find_class_with_docstring(self):
        """Test finding a class with a docstring."""
        source = '''
class DocumentedClass:
    """This is a documented class."""
    pass
'''
        result = get_class_definition(source, "DocumentedClass")

        assert result is not None
        assert result.name == "DocumentedClass"
        docstring = ast.get_docstring(result)
        assert docstring == "This is a documented class."

    def test_find_nested_class(self):
        """Test finding a nested class."""
        source = """
class OuterClass:
    class InnerClass:
        pass
"""
        result = get_class_definition(source, "InnerClass")

        assert result is not None
        assert result.name == "InnerClass"

    def test_class_not_found(self):
        """Test searching for a nonexistent class."""
        source = """
class MyClass:
    pass
"""
        result = get_class_definition(source, "NonexistentClass")

        assert result is None

    def test_multiple_classes(self):
        """Test finding the correct class among multiple classes."""
        source = """
class FirstClass:
    pass

class SecondClass:
    pass

class ThirdClass:
    pass
"""
        result = get_class_definition(source, "SecondClass")

        assert result is not None
        assert result.name == "SecondClass"

    def test_invalid_syntax(self):
        """Test parsing source with invalid syntax."""
        source = "class BrokenClass"

        with pytest.raises(SyntaxError):
            get_class_definition(source, "BrokenClass")


class TestFormatDocstringSummary:
    """Tests for format_docstring_summary function."""

    def test_format_simple_docstring(self):
        """Test formatting a simple one-line docstring."""
        result = format_docstring_summary("This is a simple docstring.")

        assert result == "This is a simple docstring."

    def test_format_multiline_docstring(self):
        """Test formatting a multi-line docstring (returns first line)."""
        docstring = """This is the first line.

This is the second paragraph.
And another line.
"""
        result = format_docstring_summary(docstring)

        assert result == "This is the first line."

    def test_format_none_docstring(self):
        """Test formatting None docstring."""
        result = format_docstring_summary(None)

        assert result == "No description available"

    def test_format_empty_docstring(self):
        """Test formatting an empty docstring."""
        result = format_docstring_summary("")

        assert result == "No description available"

    def test_format_whitespace_only_docstring(self):
        """Test formatting a docstring with only whitespace."""
        result = format_docstring_summary("   \n\n   ")

        assert result == "No description available"

    def test_format_long_docstring_truncation(self):
        """Test truncation of long docstrings."""
        long_text = "a" * 100
        result = format_docstring_summary(long_text, max_length=80)

        assert len(result) == 80
        assert result.endswith("...")
        assert result.startswith("aaa")

    def test_format_docstring_with_triple_quotes(self):
        """Test docstring that includes triple quotes gets filtered correctly."""
        # The function filters out lines starting with triple quotes
        result = format_docstring_summary('"""This is a docstring."""')

        # Since the only line starts with triple quotes, it's filtered
        assert result == "No description available"

    def test_format_docstring_preserves_short_text(self):
        """Test that short docstrings are not truncated."""
        short_text = "Short description"
        result = format_docstring_summary(short_text, max_length=80)

        assert result == short_text
        assert not result.endswith("...")

    def test_format_custom_max_length(self):
        """Test custom maximum length parameter."""
        text = "This is a moderately long description"
        result = format_docstring_summary(text, max_length=20)

        assert len(result) == 20
        assert result.endswith("...")


class TestExtractClassAttributes:
    """Tests for extract_class_attributes function."""

    def test_extract_simple_attributes(self):
        """Test extracting simple class attributes."""
        source = """
class MyClass:
    attr1 = 10
    attr2 = "hello"
    attr3 = [1, 2, 3]
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = extract_class_attributes(class_def)

        assert "attr1" in result
        assert "attr2" in result
        assert "attr3" in result

    def test_extract_annotated_attributes(self):
        """Test extracting type-annotated attributes."""
        source = """
class MyClass:
    name: str
    age: int = 25
    value: float
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = extract_class_attributes(class_def)

        assert "name" in result
        assert "age" in result
        assert "value" in result

    def test_extract_no_attributes(self):
        """Test class with no attributes."""
        source = """
class MyClass:
    def method(self):
        pass
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = extract_class_attributes(class_def)

        assert not result

    def test_extract_mixed_attributes(self):
        """Test class with both annotated and simple attributes."""
        source = """
class MyClass:
    simple = 10
    annotated: str = "test"
    no_value: int
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = extract_class_attributes(class_def)

        assert "simple" in result
        assert "annotated" in result
        assert "no_value" in result

    def test_ignores_instance_attributes(self):
        """Test that instance attributes in __init__ are not extracted."""
        source = """
class MyClass:
    class_attr = 10

    def __init__(self):
        self.instance_attr = 20
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = extract_class_attributes(class_def)

        assert "class_attr" in result
        assert "instance_attr" not in result

    def test_extract_non_name_targets(self):
        """Test handling of assignments to non-Name targets."""
        source = """
class MyClass:
    regular = 10
    a, b = 1, 2  # Tuple assignment - should be ignored
    obj.attr = 5  # Attribute assignment - should be ignored
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = extract_class_attributes(class_def)

        # Only regular should be captured
        assert "regular" in result
        assert "a" not in result
        assert "b" not in result
        # Tuple/attribute assignments are filtered out


class TestFormatClassDocstring:
    """Tests for format_class_docstring function."""

    def test_format_simple_docstring(self):
        """Test formatting a simple class docstring."""
        source = '''
class MyClass:
    """Simple docstring."""
    pass
'''
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = format_class_docstring(class_def)

        assert len(result) > 0
        assert any("Simple docstring." in line for line in result)

    def test_format_multiline_docstring(self):
        """Test formatting a multi-line class docstring."""
        source = '''
class MyClass:
    """First line.

    Second paragraph.
    """
    pass
'''
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = format_class_docstring(class_def)

        assert len(result) > 1
        joined = " ".join(result)
        assert "First line." in joined

    def test_format_no_docstring(self):
        """Test formatting when class has no docstring."""
        source = """
class MyClass:
    pass
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = format_class_docstring(class_def)

        assert not result

    def test_format_preserves_structure(self):
        """Test that docstring formatting preserves basic structure."""
        source = '''
class MyClass:
    """Docstring with formatting.

    Args:
        param1: Description

    Returns:
        Something
    """
    pass
'''
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = format_class_docstring(class_def)

        joined = "\n".join(result)
        assert "Args:" in joined
        assert "Returns:" in joined

    def test_format_empty_string_docstring(self):
        """Test formatting when docstring is an empty string."""
        # Create a ClassDef node with empty docstring manually
        # This is an edge case that's hard to create via parsing
        class_def = ast.ClassDef(
            name="TestClass",
            bases=[],
            keywords=[],
            body=[ast.Expr(value=ast.Constant(value=""))],
            decorator_list=[],
            type_params=[],
        )
        result = format_class_docstring(class_def)

        # Empty docstring should return empty list
        assert not result


class TestGetFunctionNames:
    """Tests for get_function_names function."""

    def test_get_simple_methods(self):
        """Test extracting method names from a simple class."""
        source = """
class MyClass:
    def method1(self):
        pass

    def method2(self):
        pass
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = get_function_names(class_def)

        assert "method1" in result
        assert "method2" in result
        assert len(result) == 2

    def test_get_no_methods(self):
        """Test class with no methods."""
        source = """
class MyClass:
    attr = 10
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = get_function_names(class_def)

        assert not result

    def test_get_special_methods(self):
        """Test extracting special methods like __init__."""
        source = """
class MyClass:
    def __init__(self):
        pass

    def __str__(self):
        return "MyClass"

    def regular_method(self):
        pass
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = get_function_names(class_def)

        assert "__init__" in result
        assert "__str__" in result
        assert "regular_method" in result

    def test_get_static_and_class_methods(self):
        """Test extracting static and class methods."""
        source = """
class MyClass:
    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass

    def instance_method(self):
        pass
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = get_function_names(class_def)

        assert "static_method" in result
        assert "class_method" in result
        assert "instance_method" in result

    def test_ignores_nested_functions(self):
        """Test that nested functions are not included."""
        source = """
class MyClass:
    def outer_method(self):
        def inner_function():
            pass
        return inner_function
"""
        class_def = get_class_definition(source, "MyClass")
        assert class_def is not None
        result = get_function_names(class_def)

        assert "outer_method" in result
        assert "inner_function" not in result
        assert len(result) == 1
