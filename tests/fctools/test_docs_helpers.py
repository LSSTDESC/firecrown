"""Tests for the docs_helpers module."""

import json

import pytest

from firecrown.fctools.docs_helpers import (
    load_json_file,
    validate_directory_exists,
    validate_file_exists,
    print_success,
    print_error,
)


def test_load_json_file_valid(tmp_path):
    """Test loading valid JSON file."""
    json_file = tmp_path / "valid.json"
    data = {"key": "value", "number": 42}
    json_file.write_text(json.dumps(data))

    result = load_json_file(json_file)
    assert result == data


def test_load_json_file_invalid_json(tmp_path):
    """Test JSONDecodeError is handled correctly."""
    json_file = tmp_path / "invalid.json"
    json_file.write_text("{ invalid json }")

    with pytest.raises(SystemExit) as exc_info:
        load_json_file(json_file)
    assert exc_info.value.code == 1


def test_load_json_file_read_error(tmp_path):
    """Test OSError is handled correctly."""
    json_file = tmp_path / "nonexistent.json"

    with pytest.raises(SystemExit) as exc_info:
        load_json_file(json_file)
    assert exc_info.value.code == 1


def test_validate_directory_exists_valid(tmp_path):
    """Test validating existing directory."""
    test_dir = tmp_path / "testdir"
    test_dir.mkdir()

    # Should not raise
    validate_directory_exists(test_dir)


def test_validate_directory_exists_missing(tmp_path):
    """Test validating non-existent directory."""
    test_dir = tmp_path / "nonexistent"

    with pytest.raises(SystemExit) as exc_info:
        validate_directory_exists(test_dir)
    assert exc_info.value.code == 1


def test_validate_file_exists_valid(tmp_path):
    """Test validating existing file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    # Should not raise
    validate_file_exists(test_file)


def test_validate_file_exists_missing(tmp_path):
    """Test validating non-existent file."""
    test_file = tmp_path / "nonexistent.txt"

    with pytest.raises(SystemExit) as exc_info:
        validate_file_exists(test_file)
    assert exc_info.value.code == 1


def test_print_success(capsys):
    """Test print_success outputs correctly."""
    print_success("Operation completed")
    captured = capsys.readouterr()
    assert "Operation completed" in captured.out
    assert "✓" in captured.out


def test_print_error(capsys):
    """Test print_error outputs correctly."""
    print_error("Something went wrong")
    captured = capsys.readouterr()
    assert "Something went wrong" in captured.err
    assert "✗" in captured.err
