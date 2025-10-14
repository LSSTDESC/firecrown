"""Unit tests for coverage_to_tsv timing data parsing functions.

Tests the timing data parsing functions (_parse_duration_line, _load_text_durations,
_load_json_timing, and parse_timing_data).
"""

# pylint: disable=missing-function-docstring

import json

import pytest
from rich.console import Console

from firecrown.fctools.coverage_to_tsv import (
    _load_json_timing,
    _load_text_durations,
    _parse_duration_line,
    parse_timing_data,
)

# Tests for _parse_duration_line()


def test_parse_duration_line_valid():
    line = "0.12s call tests/test_example.py::test_function"
    result = _parse_duration_line(line)
    assert result is not None
    test_name, duration = result
    assert test_name == "tests/test_example.py::test_function"
    assert duration == 0.12


def test_parse_duration_line_with_setup():
    line = "0.05s setup tests/test_example.py::test_function"
    result = _parse_duration_line(line)
    assert result is not None
    test_name, duration = result
    assert test_name == "tests/test_example.py::test_function"
    assert duration == 0.05


def test_parse_duration_line_with_teardown():
    line = "0.03s teardown tests/test_example.py::test_function"
    result = _parse_duration_line(line)
    assert result is not None
    test_name, duration = result
    assert test_name == "tests/test_example.py::test_function"
    assert duration == 0.03


def test_parse_duration_line_integer_duration():
    line = "1s call tests/test_example.py::test_function"
    result = _parse_duration_line(line)
    assert result is not None
    test_name, duration = result
    assert test_name == "tests/test_example.py::test_function"
    assert duration == 1.0


def test_parse_duration_line_no_phase():
    line = "0.25s tests/test_example.py::test_function"
    result = _parse_duration_line(line)
    assert result is not None
    test_name, duration = result
    assert test_name == "tests/test_example.py::test_function"
    assert duration == 0.25


def test_parse_duration_line_invalid():
    line = "This is not a valid duration line"
    result = _parse_duration_line(line)
    assert result is None


def test_parse_duration_line_no_duration():
    line = "tests/test_example.py::test_function"
    result = _parse_duration_line(line)
    assert result is None


def test_parse_duration_line_invalid_duration_format():
    line = "abc call tests/test_example.py::test_function"
    result = _parse_duration_line(line)
    assert result is None


# Tests for _load_text_durations()


def test_load_text_durations(tmp_path):
    timing_file = tmp_path / "durations.txt"
    timing_file.write_text(
        "0.12s call tests/test_example.py::test_func1\n"
        "0.05s setup tests/test_example.py::test_func1\n"
        "0.08s call tests/test_example.py::test_func2\n"
        "Invalid line here\n"
        "0.03s teardown tests/test_example.py::test_func1\n"
    )

    result = _load_text_durations(timing_file)

    # test_func1 should have combined setup + call + teardown
    expected_duration = 0.12 + 0.05 + 0.03
    assert result["tests/test_example.py::test_func1"] == pytest.approx(
        expected_duration
    )
    assert result["tests/test_example.py::test_func2"] == pytest.approx(0.08)


def test_load_text_durations_empty_file(tmp_path):
    timing_file = tmp_path / "empty.txt"
    timing_file.write_text("")

    result = _load_text_durations(timing_file)
    assert not result


def test_load_text_durations_all_invalid(tmp_path):
    timing_file = tmp_path / "invalid.txt"
    timing_file.write_text("Invalid line 1\nInvalid line 2\nInvalid line 3\n")

    result = _load_text_durations(timing_file)
    assert not result


# Tests for _load_json_timing()


def test_load_json_timing(tmp_path):
    timing_file = tmp_path / "timing.json"
    data = {
        "tests": [
            {"nodeid": "tests/test_a.py::test_one", "duration": 0.5},
            {"nodeid": "tests/test_a.py::test_two", "duration": 1.2},
            {"nodeid": "tests/test_b.py::test_three", "duration": 0.3},
        ]
    }
    timing_file.write_text(json.dumps(data))

    result = _load_json_timing(timing_file)

    assert result["tests/test_a.py::test_one"] == 0.5
    assert result["tests/test_a.py::test_two"] == 1.2
    assert result["tests/test_b.py::test_three"] == 0.3


def test_load_json_timing_missing_fields(tmp_path):
    timing_file = tmp_path / "timing.json"
    data = {
        "tests": [
            {"nodeid": "tests/test_a.py::test_one"},  # No duration
            {"duration": 0.5},  # No nodeid
            {"nodeid": "tests/test_b.py::test_two", "duration": None},  # Null duration
        ]
    }
    timing_file.write_text(json.dumps(data))

    result = _load_json_timing(timing_file)

    assert result["tests/test_a.py::test_one"] == 0.0
    assert result[""] == 0.5
    assert result["tests/test_b.py::test_two"] == 0.0


def test_load_json_timing_invalid_duration(tmp_path):
    timing_file = tmp_path / "timing.json"
    data = {
        "tests": [
            {"nodeid": "tests/test_a.py::test_one", "duration": "invalid"},
        ]
    }
    timing_file.write_text(json.dumps(data))

    result = _load_json_timing(timing_file)

    assert result["tests/test_a.py::test_one"] == 0.0


def test_load_json_timing_no_tests_key(tmp_path):
    timing_file = tmp_path / "timing.json"
    data = {"summary": "some data", "other_field": []}
    timing_file.write_text(json.dumps(data))

    result = _load_json_timing(timing_file)

    assert not result


# Tests for parse_timing_data()


def test_parse_timing_data_json_format(tmp_path):
    console = Console()
    timing_file = tmp_path / "timing.json"
    data = {
        "tests": [
            {"nodeid": "tests/test_a.py::test_one", "duration": 0.5},
        ]
    }
    timing_file.write_text(json.dumps(data))

    result = parse_timing_data(console, timing_file)

    assert result["tests/test_a.py::test_one"] == 0.5


def test_parse_timing_data_text_format(tmp_path):
    console = Console()
    timing_file = tmp_path / "durations.txt"
    timing_file.write_text("0.12s call tests/test_example.py::test_func\n")

    result = parse_timing_data(console, timing_file)

    assert result["tests/test_example.py::test_func"] == pytest.approx(0.12)


def test_parse_timing_data_none():
    console = Console()
    result = parse_timing_data(console, None)
    assert not result


def test_parse_timing_data_nonexistent_file(tmp_path):
    console = Console()
    timing_file = tmp_path / "nonexistent.json"
    result = parse_timing_data(console, timing_file)
    assert not result


def test_parse_timing_data_json_decode_error_fallback(tmp_path):
    console = Console()
    # File that's not valid JSON but has valid text durations
    timing_file = tmp_path / "mixed.txt"
    timing_file.write_text("0.15s call tests/test_example.py::test_func\n")

    result = parse_timing_data(console, timing_file)

    # Should fall back to text parsing
    assert result["tests/test_example.py::test_func"] == pytest.approx(0.15)


def test_parse_timing_data_oserror_during_text_read(tmp_path, capsys, monkeypatch):
    console = Console()
    # Create a file that will cause JSON parse error
    timing_file = tmp_path / "bad.txt"
    timing_file.write_text("not json")

    # Mock _load_text_durations to raise OSError
    def mock_load_text(*args, **kwargs):
        raise OSError("Mocked error")

    monkeypatch.setattr(
        "firecrown.fctools.coverage_to_tsv._load_text_durations", mock_load_text
    )

    result = parse_timing_data(console, timing_file)

    assert not result
    captured = capsys.readouterr()
    assert "Warning: Could not parse timing data" in captured.out
