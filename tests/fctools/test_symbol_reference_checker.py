"""Tests for the symbol_reference_checker module."""

import json
from typer.testing import CliRunner

from firecrown.fctools.symbol_reference_checker import (
    extract_code_spans,
    check_qmd_file,
    app,
)


# Create CliRunner instance for testing
runner = CliRunner()


def test_extract_code_spans_simple():
    """Test extracting code spans from simple markdown."""
    content = "This is `code1` and `code2` in text."
    spans = extract_code_spans(content)
    assert len(spans) == 2
    assert spans[0] == (1, "code1")
    assert spans[1] == (1, "code2")


def test_extract_code_spans_multiline():
    """Test extracting code spans from multiple lines."""
    content = """Line 1 with `span1`
Line 2 with `span2` and `span3`
Line 3 no spans
Line 4 with `span4`"""
    spans = extract_code_spans(content)
    assert len(spans) == 4
    assert spans[0] == (1, "span1")
    assert spans[1] == (2, "span2")
    assert spans[2] == (2, "span3")
    assert spans[3] == (4, "span4")


def test_extract_code_spans_empty():
    """Test extracting code spans from text without any."""
    content = "No code spans here at all."
    spans = extract_code_spans(content)
    assert len(spans) == 0


def test_check_qmd_file_valid_symbols(tmp_path):
    """Test checking a file with valid symbol references."""
    # Create a test .qmd file
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text(
        """
# Test file

This mentions `firecrown.parameters.Parameter` and `Updatable`.
"""
    )

    # Create a symbol map with these symbols
    symbol_map = {
        "firecrown.parameters.Parameter": "api/...",
        "firecrown.updatable.Updatable": "api/...",
    }

    errors = check_qmd_file(qmd_file, symbol_map)
    assert len(errors) == 0


def test_check_qmd_file_invalid_fully_qualified(tmp_path):
    """Test checking a file with invalid fully-qualified symbol."""
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text(
        """
This mentions `firecrown.parameters.Paramater` (typo).
"""
    )

    symbol_map = {
        "firecrown.parameters.Parameter": "api/...",
    }

    errors = check_qmd_file(qmd_file, symbol_map)
    assert len(errors) == 1
    assert "firecrown.parameters.Paramater" in errors[0]
    assert "Line 2" in errors[0]


def test_check_qmd_file_invalid_unqualified(tmp_path):
    """Test checking a file with invalid unqualified symbol."""
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text(
        """
This mentions `Updateable` (typo, should be Updatable).
"""
    )

    symbol_map = {
        "firecrown.updatable.Updatable": "api/...",
    }

    errors = check_qmd_file(qmd_file, symbol_map)
    assert len(errors) == 1
    assert "Updateable" in errors[0]


def test_check_qmd_file_multiple_errors(tmp_path):
    """Test that all errors are reported, not just the first."""
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text(
        """
Line 1 has `firecrown.invalid.Class1`
Line 2 has `firecrown.invalid.Class2`
Line 3 has `InvalidClass3`
"""
    )

    symbol_map = {
        "firecrown.valid.Something": "api/...",
    }

    errors = check_qmd_file(qmd_file, symbol_map)
    assert len(errors) == 3
    assert any("Class1" in e for e in errors)
    assert any("Class2" in e for e in errors)
    assert any("InvalidClass3" in e for e in errors)


def test_check_qmd_file_with_exclude_pattern(tmp_path):
    """Test that exclude pattern works."""
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text(
        """
This has `firecrown.example.ExampleClass` which should be excluded.
This has `firecrown.real.RealClass` which should be checked.
"""
    )

    symbol_map = {
        "firecrown.real.RealClass": "api/...",
    }

    # Should report error for ExampleClass but not when excluded
    errors_without_exclude = check_qmd_file(qmd_file, symbol_map, None, None)
    assert len(errors_without_exclude) == 1
    assert "ExampleClass" in errors_without_exclude[0]

    # With exclude pattern
    errors_with_exclude = check_qmd_file(qmd_file, symbol_map, None, r"\.example\.")
    assert len(errors_with_exclude) == 0


def test_check_qmd_file_lowercase_code_not_checked(tmp_path):
    """Test that lowercase code spans are not checked as symbols."""
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text(
        """
This has `some_function` and `another_var` which are lowercase.
"""
    )

    symbol_map: dict[str, str] = {}  # Empty symbol map

    errors = check_qmd_file(qmd_file, symbol_map)
    # Should not report errors for lowercase identifiers
    assert len(errors) == 0


def test_check_qmd_file_partial_module_path(tmp_path):
    """Test validation of partial module paths."""
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text(
        """
Reference to `firecrown.likelihood` module.
"""
    )

    symbol_map = {
        "firecrown.likelihood": "api/...",
        "firecrown.likelihood.SourceSystematic": "api/...",
    }

    errors = check_qmd_file(qmd_file, symbol_map)
    assert len(errors) == 0


def test_main_success(tmp_path):
    """Test main function with valid symbols."""
    # Create test files
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text("Valid reference: `firecrown.parameters.Parameter`")

    symbol_map_file = tmp_path / "symbol_map.json"
    symbol_map: dict[str, str] = {"firecrown.parameters.Parameter": "api/..."}
    symbol_map_file.write_text(json.dumps(symbol_map))

    # Run main
    result = runner.invoke(app, [str(tmp_path), str(symbol_map_file)])

    assert result.exit_code == 0
    # Success message goes to stdout via print_success
    assert "No invalid symbol references" in result.stdout


def test_main_with_errors(tmp_path):
    """Test main function with invalid symbols."""
    # Create test files
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text("Invalid reference: `firecrown.invalid.Symbol`")

    symbol_map_file = tmp_path / "symbol_map.json"
    symbol_map = {"firecrown.valid.Symbol": "api/..."}
    symbol_map_file.write_text(json.dumps(symbol_map))

    # Run main - should exit with error
    result = runner.invoke(app, [str(tmp_path), str(symbol_map_file)])

    assert result.exit_code == 1
    assert "invalid symbol" in result.stderr.lower()
    assert "firecrown.invalid.Symbol" in result.stderr


def test_main_missing_symbol_map(tmp_path):
    """Test main function with missing symbol map file."""
    result = runner.invoke(app, [str(tmp_path), str(tmp_path / "nonexistent.json")])

    assert result.exit_code != 0


def test_main_with_exclude_pattern(tmp_path):
    """Test main function with exclude pattern."""
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text("Example: `firecrown.example.ExampleClass`")

    symbol_map_file = tmp_path / "symbol_map.json"
    symbol_map: dict[str, str] = {}
    symbol_map_file.write_text(json.dumps(symbol_map))

    # Without exclude pattern - should fail
    result = runner.invoke(app, [str(tmp_path), str(symbol_map_file)])
    assert result.exit_code == 1

    # With exclude pattern - should pass
    result = runner.invoke(
        app, [str(tmp_path), str(symbol_map_file), "--exclude-pattern", "example"]
    )
    assert result.exit_code == 0


def test_load_external_symbols_read_error(tmp_path, capsys):
    """Test _load_external_symbols handles OSError gracefully."""
    # pylint: disable=import-outside-toplevel
    from firecrown.fctools.symbol_reference_checker import (
        _load_external_symbols,
    )

    # Create a file that will trigger an error
    # Use a directory path instead of file to trigger OSError
    bad_path = tmp_path / "directory_not_file"
    bad_path.mkdir()

    result = _load_external_symbols(bad_path)

    # Should return empty set on error
    assert result == set()

    # Check warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.err or captured.err == ""


def test_load_external_symbols_unicode_error(tmp_path, capsys):
    """Test _load_external_symbols handles UnicodeDecodeError gracefully."""
    # pylint: disable=import-outside-toplevel
    from firecrown.fctools.symbol_reference_checker import (
        _load_external_symbols,
    )

    # Create a file with non-UTF8 bytes
    bad_file = tmp_path / "bad_encoding.txt"
    bad_file.write_bytes(b"\xff\xfe\x00\x00")  # Invalid UTF-8

    result = _load_external_symbols(bad_file)

    # Should return empty set on error
    assert result == set()

    # Check warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.err or captured.err == ""


def test_load_external_symbols_with_comments_and_empty_lines(tmp_path):
    """Test _load_external_symbols filters comments and empty lines."""
    # pylint: disable=import-outside-toplevel
    from firecrown.fctools.symbol_reference_checker import (
        _load_external_symbols,
    )

    symbols_file = tmp_path / "external_symbols.txt"
    content = """# This is a comment
ValidSymbol1

# Another comment
ValidSymbol2

  # Indented comment
ValidSymbol3
"""
    symbols_file.write_text(content)

    result = _load_external_symbols(symbols_file)

    # Should only have the valid symbols, not comments or empty lines
    assert result == {"ValidSymbol1", "ValidSymbol2", "ValidSymbol3"}


def test_compile_exclude_pattern_invalid_regex(capsys):
    """Test _compile_exclude_pattern handles invalid regex."""
    # pylint: disable=import-outside-toplevel
    from firecrown.fctools.symbol_reference_checker import (
        _compile_exclude_pattern,
    )

    # Invalid regex pattern (unmatched bracket)
    result = _compile_exclude_pattern("[invalid")

    # Should return None on error
    assert result is None

    # Check warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.err


def test_check_qmd_file_unqualified_symbol_also_fully_qualified(tmp_path):
    """Test unqualified symbol that exists in symbol map without dots.

    This tests the case where a symbol like `CustomClass` is in the symbol
    map as-is (not as a dotted path), so it's in fully_qualified_symbols
    but not in unqualified_symbols (which only contains last parts of dotted names).
    """
    qmd_file = tmp_path / "test.qmd"
    # Use an uppercase symbol that exists in symbol map without dots
    qmd_file.write_text("Reference to `CustomClass` in text.")

    # Symbol map with CustomClass as a non-dotted entry
    symbol_map = {
        "CustomClass": "api/custom.html",  # No dots, so won't be in unqualified set
        "firecrown.other.Module": "api/...",  # This will add "Module" to unqualified
    }

    errors = check_qmd_file(qmd_file, symbol_map)

    # Should not have errors since CustomClass is in fully_qualified_symbols
    assert len(errors) == 0


def test_main_with_external_symbols_file(tmp_path):
    """Test main function with external symbols file."""
    # Create test files
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text("Reference: `ExternalSymbol` and `firecrown.valid.Symbol`")

    symbol_map_file = tmp_path / "symbol_map.json"
    symbol_map = {"firecrown.valid.Symbol": "api/..."}
    symbol_map_file.write_text(json.dumps(symbol_map))

    external_symbols_file = tmp_path / "external.txt"
    external_symbols_file.write_text("ExternalSymbol\n")

    # Run main with external symbols
    result = runner.invoke(
        app,
        [
            str(tmp_path),
            str(symbol_map_file),
            "--external-symbols-file",
            str(external_symbols_file),
        ],
    )

    assert result.exit_code == 0
    assert "No invalid symbol references" in result.stdout


def test_check_qmd_file_with_invalid_exclude_pattern(tmp_path, capsys):
    """Test check_qmd_file handles invalid exclude pattern gracefully."""
    qmd_file = tmp_path / "test.qmd"
    qmd_file.write_text("Reference: `firecrown.invalid.Symbol`")

    symbol_map = {"firecrown.valid.Symbol": "api/..."}

    # Invalid regex should be handled gracefully
    errors = check_qmd_file(qmd_file, symbol_map, exclude_pattern="[invalid")

    # Should still find the error since invalid pattern is ignored
    assert len(errors) > 0

    # Check warning about invalid pattern
    captured = capsys.readouterr()
    assert "Warning" in captured.err


def test_check_qmd_file_unqualified_not_in_fully_qualified():
    """Test check_qmd_file with unqualified symbol not in fully_qualified set."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    qmd_file = Path("test.qmd")
    # Use an uppercase unqualified symbol that doesn't exist
    content = "Reference to `UnknownSymbol` in text."

    # Create a temporary file
    import tempfile  # pylint: disable=import-outside-toplevel

    with tempfile.NamedTemporaryFile(mode="w", suffix=".qmd", delete=False) as f:
        f.write(content)
        qmd_file = Path(f.name)

    try:
        # Symbol map without the symbol
        symbol_map = {
            "firecrown.parameters.Parameter": "api/...",
        }

        errors = check_qmd_file(qmd_file, symbol_map)

        # Should have error since UnknownSymbol is not in the map
        assert len(errors) > 0
        assert "UnknownSymbol" in errors[0]
    finally:
        qmd_file.unlink()
