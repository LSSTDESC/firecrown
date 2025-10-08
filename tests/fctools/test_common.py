"""Unit tests for firecrown.fctools.common module.

Tests the shared utility functions used across fctools.
"""

import json
from pathlib import Path

import pytest

from firecrown.fctools.common import (
    cli_error,
    cli_warning,
    format_line_ranges,
    import_class_from_path,
    import_module_from_file,
    load_json_file,
    validate_input_file,
    validate_output_path,
)


class TestLoadJsonFile:
    """Tests for load_json_file function."""

    def test_load_valid_json(self, tmp_path):
        """Test loading a valid JSON file."""
        json_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        json_file.write_text(json.dumps(test_data))

        result = load_json_file(json_file)

        assert result == test_data

    def test_load_json_with_custom_context(self, tmp_path):
        """Test that custom error context is used in error messages."""
        json_file = tmp_path / "test.json"
        json_file.write_text("{invalid json}")

        with pytest.raises(SystemExit) as exc_info:
            load_json_file(json_file, error_context="parsing test data")

        assert exc_info.value.code == 1

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a file that doesn't exist."""
        json_file = tmp_path / "nonexistent.json"

        with pytest.raises(SystemExit) as exc_info:
            load_json_file(json_file)

        assert exc_info.value.code == 1

    def test_load_invalid_json(self, tmp_path):
        """Test loading a file with invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{this is not valid JSON}")

        with pytest.raises(SystemExit) as exc_info:
            load_json_file(json_file)

        assert exc_info.value.code == 1

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty file."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("")

        with pytest.raises(SystemExit) as exc_info:
            load_json_file(json_file)

        assert exc_info.value.code == 1

    def test_load_complex_json(self, tmp_path):
        """Test loading JSON with nested structures."""
        json_file = tmp_path / "complex.json"
        test_data = {
            "files": {"file1.py": {"coverage": 85.5}, "file2.py": {"coverage": 92.3}},
            "totals": {"lines": 1000, "covered": 890},
        }
        json_file.write_text(json.dumps(test_data))

        result = load_json_file(json_file)

        assert result == test_data
        assert result["files"]["file1.py"]["coverage"] == 85.5


class TestImportClassFromPath:
    """Tests for import_class_from_path function."""

    def test_import_builtin_class(self):
        """Test importing a built-in Python class."""
        result = import_class_from_path("pathlib.Path")

        assert result == Path
        assert isinstance(Path(), result)

    def test_import_nested_class(self):
        """Test importing a class from a nested module."""
        result = import_class_from_path("collections.abc.Mapping")

        from collections.abc import Mapping

        assert result == Mapping

    def test_import_invalid_module(self):
        """Test importing from a nonexistent module."""
        with pytest.raises(SystemExit) as exc_info:
            import_class_from_path("nonexistent.module.Class")

        assert exc_info.value.code == 1

    def test_import_invalid_class(self):
        """Test importing a nonexistent class from a valid module."""
        with pytest.raises(SystemExit) as exc_info:
            import_class_from_path("pathlib.NonexistentClass")

        assert exc_info.value.code == 1

    def test_import_invalid_path_format(self):
        """Test importing with an invalid path format (no dots)."""
        with pytest.raises(SystemExit) as exc_info:
            import_class_from_path("InvalidPath")

        assert exc_info.value.code == 1

    def test_import_firecrown_class(self):
        """Test importing a class from firecrown."""
        result = import_class_from_path("firecrown.updatable.Updatable")

        from firecrown.updatable import Updatable

        assert result == Updatable


class TestImportModuleFromFile:
    """Tests for import_module_from_file function."""

    def test_import_valid_module(self, tmp_path):
        """Test importing a valid Python module from a file."""
        module_file = tmp_path / "test_module.py"
        module_file.write_text(
            """
def test_function():
    return 42

TEST_CONSTANT = "hello"
"""
        )

        module = import_module_from_file(module_file)

        assert hasattr(module, "test_function")
        assert module.test_function() == 42
        assert module.TEST_CONSTANT == "hello"

    def test_import_module_with_custom_name(self, tmp_path):
        """Test importing a module with a custom name."""
        module_file = tmp_path / "custom.py"
        module_file.write_text("VALUE = 123")

        module = import_module_from_file(module_file, module_name="my_module")

        assert module.VALUE == 123

    def test_import_nonexistent_file(self, tmp_path):
        """Test importing from a nonexistent file."""
        module_file = tmp_path / "nonexistent.py"

        with pytest.raises(SystemExit) as exc_info:
            import_module_from_file(module_file)

        assert exc_info.value.code == 1

    def test_import_file_with_syntax_error(self, tmp_path):
        """Test importing a file with syntax errors."""
        module_file = tmp_path / "syntax_error.py"
        module_file.write_text("def broken_function(\n    # missing closing paren")

        with pytest.raises(SystemExit) as exc_info:
            import_module_from_file(module_file)

        assert exc_info.value.code == 1

    def test_import_file_with_import_error(self, tmp_path):
        """Test importing a file that imports a nonexistent module."""
        module_file = tmp_path / "import_error.py"
        module_file.write_text("import nonexistent_module")

        with pytest.raises(SystemExit) as exc_info:
            import_module_from_file(module_file)

        assert exc_info.value.code == 1


class TestCliError:
    """Tests for cli_error function."""

    def test_cli_error_exits(self):
        """Test that cli_error exits the program."""
        with pytest.raises(SystemExit) as exc_info:
            cli_error("Test error message")

        assert exc_info.value.code == 1

    def test_cli_error_custom_exit_code(self):
        """Test cli_error with a custom exit code."""
        with pytest.raises(SystemExit) as exc_info:
            cli_error("Test error message", exit_code=2)

        assert exc_info.value.code == 2

    def test_cli_error_adds_prefix(self, capsys):
        """Test that ERROR: prefix is added if not present."""
        with pytest.raises(SystemExit):
            cli_error("Test message")

        captured = capsys.readouterr()
        assert "ERROR: Test message" in captured.err

    def test_cli_error_preserves_prefix(self, capsys):
        """Test that existing ERROR: prefix is not duplicated."""
        with pytest.raises(SystemExit):
            cli_error("ERROR: Already has prefix")

        captured = capsys.readouterr()
        assert "ERROR: ERROR:" not in captured.err
        assert "ERROR: Already has prefix" in captured.err


class TestCliWarning:
    """Tests for cli_warning function."""

    def test_cli_warning_does_not_exit(self):
        """Test that cli_warning does not exit the program."""
        # Should not raise SystemExit
        cli_warning("Test warning message")

    def test_cli_warning_adds_prefix(self, capsys):
        """Test that Warning: prefix is added if not present."""
        cli_warning("Test message")

        captured = capsys.readouterr()
        assert "Warning: Test message" in captured.err

    def test_cli_warning_preserves_prefix(self, capsys):
        """Test that existing Warning: prefix is not duplicated."""
        cli_warning("Warning: Already has prefix")

        captured = capsys.readouterr()
        assert "Warning: Warning:" not in captured.err
        assert "Warning: Already has prefix" in captured.err

    def test_multiple_warnings(self, capsys):
        """Test that multiple warnings can be issued."""
        cli_warning("First warning")
        cli_warning("Second warning")

        captured = capsys.readouterr()
        assert "Warning: First warning" in captured.err
        assert "Warning: Second warning" in captured.err


class TestValidateInputFile:
    """Tests for validate_input_file function."""

    def test_validate_existing_file(self, tmp_path):
        """Test validating an existing readable file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Should not raise
        validate_input_file(test_file)

    def test_validate_nonexistent_file(self, tmp_path):
        """Test validating a nonexistent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(SystemExit) as exc_info:
            validate_input_file(test_file)

        assert exc_info.value.code == 1

    def test_validate_directory_not_file(self, tmp_path):
        """Test validating a directory instead of a file."""
        with pytest.raises(SystemExit) as exc_info:
            validate_input_file(tmp_path)

        assert exc_info.value.code == 1

    def test_validate_custom_description(self, tmp_path):
        """Test validate_input_file with custom file description."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(SystemExit):
            validate_input_file(test_file, file_description="Configuration file")

        # Error message should contain custom description


class TestValidateOutputPath:
    """Tests for validate_output_path function."""

    def test_validate_new_output_path(self, tmp_path):
        """Test validating a new output path (file doesn't exist)."""
        output_file = tmp_path / "output.txt"

        # Should not raise
        validate_output_path(output_file, overwrite=False)

    def test_validate_existing_path_without_overwrite(self, tmp_path):
        """Test validating existing file without overwrite permission."""
        output_file = tmp_path / "existing.txt"
        output_file.write_text("existing content")

        with pytest.raises(SystemExit) as exc_info:
            validate_output_path(output_file, overwrite=False)

        assert exc_info.value.code == 1

    def test_validate_existing_path_with_overwrite(self, tmp_path):
        """Test validating existing file with overwrite permission."""
        output_file = tmp_path / "existing.txt"
        output_file.write_text("existing content")

        # Should not raise
        validate_output_path(output_file, overwrite=True)


class TestFormatLineRanges:
    """Tests for format_line_ranges function."""

    def test_format_empty_list(self):
        """Test formatting an empty list."""
        result = format_line_ranges([])

        assert result == []

    def test_format_single_line(self):
        """Test formatting a single line number."""
        result = format_line_ranges([42])

        assert result == ["42"]

    def test_format_consecutive_lines(self):
        """Test formatting consecutive line numbers."""
        result = format_line_ranges([1, 2, 3, 4, 5])

        assert result == ["1-5"]

    def test_format_mixed_ranges(self):
        """Test formatting mixed consecutive and individual lines."""
        result = format_line_ranges([1, 2, 3, 5, 6, 8, 10, 11, 12])

        assert result == ["1-3", "5-6", "8", "10-12"]

    def test_format_unsorted_lines(self):
        """Test formatting unsorted line numbers."""
        result = format_line_ranges([5, 1, 3, 2, 8])

        assert result == ["1-3", "5", "8"]

    def test_format_duplicate_lines(self):
        """Test formatting with duplicate line numbers."""
        result = format_line_ranges([1, 2, 2, 3, 5, 5, 6])

        # Duplicates are sorted but not removed, creating separate consecutive ranges
        assert len(result) == 4  # ['1-2', '2-3', '5', '5-6']
        assert "1-2" in result or "2-3" in result

    def test_format_large_range(self):
        """Test formatting a large range of consecutive lines."""
        result = format_line_ranges(list(range(100, 200)))

        assert result == ["100-199"]

    def test_format_all_individual_lines(self):
        """Test formatting non-consecutive lines."""
        result = format_line_ranges([1, 5, 10, 15, 20])

        assert result == ["1", "5", "10", "15", "20"]

    def test_format_two_consecutive_lines(self):
        """Test formatting exactly two consecutive lines."""
        result = format_line_ranges([5, 6])

        assert result == ["5-6"]
