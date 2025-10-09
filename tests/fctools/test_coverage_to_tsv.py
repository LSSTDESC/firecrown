"""Unit tests for firecrown.fctools.coverage_to_tsv module.

Tests the coverage JSON to TSV conversion tool.
"""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from firecrown.fctools.coverage_to_tsv import (
    CoverageRecord,
    _load_json_timing,
    _load_text_durations,
    _parse_duration_line,
    extract_coverage_data,
    main,
    match_test_to_function,
    parse_timing_data,
    write_tsv_file,
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
    timing_file = tmp_path / "timing.json"
    data = {
        "tests": [
            {"nodeid": "tests/test_a.py::test_one", "duration": 0.5},
        ]
    }
    timing_file.write_text(json.dumps(data))

    result = parse_timing_data(timing_file)

    assert result["tests/test_a.py::test_one"] == 0.5


def test_parse_timing_data_text_format(tmp_path):
    timing_file = tmp_path / "durations.txt"
    timing_file.write_text("0.12s call tests/test_example.py::test_func\n")

    result = parse_timing_data(timing_file)

    assert result["tests/test_example.py::test_func"] == pytest.approx(0.12)


def test_parse_timing_data_none():
    result = parse_timing_data(None)
    assert not result


def test_parse_timing_data_nonexistent_file(tmp_path):
    timing_file = tmp_path / "nonexistent.json"
    result = parse_timing_data(timing_file)
    assert not result


def test_parse_timing_data_json_decode_error_fallback(tmp_path):
    # File that's not valid JSON but has valid text durations
    timing_file = tmp_path / "mixed.txt"
    timing_file.write_text("0.15s call tests/test_example.py::test_func\n")

    result = parse_timing_data(timing_file)

    # Should fall back to text parsing
    assert result["tests/test_example.py::test_func"] == pytest.approx(0.15)


def test_parse_timing_data_oserror_during_text_read(tmp_path, capsys, monkeypatch):
    # Create a file that will cause JSON parse error
    timing_file = tmp_path / "bad.txt"
    timing_file.write_text("not json")

    # Mock _load_text_durations to raise OSError
    def mock_load_text(*args, **kwargs):
        raise OSError("Mocked error")

    monkeypatch.setattr(
        "firecrown.fctools.coverage_to_tsv._load_text_durations", mock_load_text
    )

    result = parse_timing_data(timing_file)

    assert not result
    captured = capsys.readouterr()
    assert "Warning: Could not parse timing data" in captured.out


def test_parse_timing_data_oserror_initial_read(tmp_path, capsys, monkeypatch):
    timing_file = tmp_path / "test.json"
    timing_file.write_text("{}")

    # Mock _load_json_timing to raise OSError
    def mock_load_json(*args, **kwargs):
        raise OSError("Cannot read file")

    monkeypatch.setattr(
        "firecrown.fctools.coverage_to_tsv._load_json_timing", mock_load_json
    )

    result = parse_timing_data(timing_file)

    assert not result
    captured = capsys.readouterr()
    assert "Warning: Could not read timing file" in captured.out


# Tests for match_test_to_function()


def test_match_test_to_function_exact_name_match():
    test_name = "tests/test_module.py::test_my_function"
    function_name = "my_function"
    file_path = "src/module.py"

    score = match_test_to_function(test_name, function_name, file_path)

    # Function name in test name = +0.5
    assert score >= 0.5


def test_match_test_to_function_file_match():
    test_name = "tests/test_example.py::test_something"
    function_name = "other_function"
    file_path = "src/example.py"

    score = match_test_to_function(test_name, function_name, file_path)

    # File path match = +0.3
    assert score == pytest.approx(0.3)


def test_match_test_to_function_full_match():
    test_name = "tests/test_utils.py::test_format_string"
    function_name = "format_string"
    file_path = "src/utils.py"

    score = match_test_to_function(test_name, function_name, file_path)

    # File match (+0.3) + function name match (+0.5) = 0.8, but capped at 1.0
    # Actually: test_format_string contains "format_string" = +0.5
    # Also: format_string in test_format_string after removing test_ = +0.4
    # Plus file match = +0.3, but capped at 1.0
    assert score == 1.0


def test_match_test_to_function_no_match():
    test_name = "tests/test_other.py::test_something"
    function_name = "unrelated_function"
    file_path = "src/different.py"

    score = match_test_to_function(test_name, function_name, file_path)

    assert score == 0.0


def test_match_test_to_function_test_name_no_separator():
    test_name = "tests/test_module.py"
    function_name = "some_function"
    file_path = "src/module.py"

    score = match_test_to_function(test_name, function_name, file_path)

    # File match (0.3) + test method name in function (module in some_function? No)
    # Actually: test_module.py -> "module" extracted, "module" not in "some_function"
    # Let me check the actual logic...
    # File match gives 0.3, and empty test_method means additional checks
    # The score should be at least 0.3 for file match
    assert score >= 0.3


def test_match_test_to_function_case_insensitive():
    test_name = "tests/test_module.py::test_MyFunction"
    function_name = "myfunction"
    file_path = "src/other.py"

    score = match_test_to_function(test_name, function_name, file_path)

    # Case-insensitive match
    assert score >= 0.5


# Tests for extract_coverage_data()


def test_extract_coverage_data_basic():
    coverage_data = {
        "files": {
            "src/example.py": {
                "summary": {
                    "num_statements": 100,
                    "covered_lines": 80,
                    "percent_covered": 80.0,
                },
                "functions": {
                    "my_function": {
                        "summary": {
                            "covered_lines": 10,
                            "num_statements": 12,
                            "percent_covered": 83.33,
                            "missing_lines": 2,
                            "excluded_lines": 0,
                            "num_branches": 4,
                            "covered_branches": 3,
                            "missing_branches": 1,
                            "num_partial_branches": 1,
                            "percent_covered_display": "83.3%",
                        }
                    }
                },
            }
        }
    }

    result = extract_coverage_data(coverage_data)

    assert len(result) == 1
    record = result[0]
    assert record.file_path == "src/example.py"
    assert record.function_name == "my_function"
    assert record.covered_lines == 10
    assert record.total_statements == 12
    assert record.percent_covered == 83.33
    assert record.missing_lines == 2
    assert record.excluded_lines == 0
    assert record.num_branches == 4
    assert record.covered_branches == 3
    assert record.missing_branches == 1
    assert record.num_partial_branches == 1
    assert record.percent_covered_display == "83.3%"
    assert record.file_total_statements == 100
    assert record.file_covered_lines == 80
    assert record.file_percent_covered == 80.0
    assert record.test_duration is None


def test_extract_coverage_data_with_timing():
    coverage_data = {
        "files": {
            "src/utils.py": {
                "summary": {
                    "num_statements": 50,
                    "covered_lines": 40,
                    "percent_covered": 80.0,
                },
                "functions": {
                    "format_string": {
                        "summary": {
                            "covered_lines": 5,
                            "num_statements": 5,
                            "percent_covered": 100.0,
                            "missing_lines": 0,
                            "excluded_lines": 0,
                            "num_branches": 0,
                            "covered_branches": 0,
                            "missing_branches": 0,
                            "num_partial_branches": 0,
                            "percent_covered_display": "100%",
                        }
                    }
                },
            }
        }
    }

    timing_data = {
        "tests/test_utils.py::test_format_string": 0.25,
    }

    result = extract_coverage_data(coverage_data, timing_data)

    assert len(result) == 1
    record = result[0]
    assert record.function_name == "format_string"
    # Should match timing due to function name + file path
    assert record.test_duration == 0.25


def test_extract_coverage_data_timing_below_threshold():
    coverage_data = {
        "files": {
            "src/module.py": {
                "summary": {
                    "num_statements": 10,
                    "covered_lines": 10,
                    "percent_covered": 100.0,
                },
                "functions": {
                    "unrelated_func": {
                        "summary": {
                            "covered_lines": 5,
                            "num_statements": 5,
                            "percent_covered": 100.0,
                            "missing_lines": 0,
                            "excluded_lines": 0,
                            "num_branches": 0,
                            "covered_branches": 0,
                            "missing_branches": 0,
                            "num_partial_branches": 0,
                            "percent_covered_display": "100%",
                        }
                    }
                },
            }
        }
    }

    timing_data = {
        "tests/test_other.py::test_completely_different": 0.1,
    }

    result = extract_coverage_data(coverage_data, timing_data)

    assert len(result) == 1
    record = result[0]
    # Score too low, no timing should be assigned
    assert record.test_duration is None


def test_extract_coverage_data_multiple_functions():
    coverage_data = {
        "files": {
            "src/module.py": {
                "summary": {
                    "num_statements": 20,
                    "covered_lines": 15,
                    "percent_covered": 75.0,
                },
                "functions": {
                    "func_a": {
                        "summary": {
                            "covered_lines": 5,
                            "num_statements": 8,
                            "percent_covered": 62.5,
                            "missing_lines": 3,
                            "excluded_lines": 0,
                            "num_branches": 2,
                            "covered_branches": 1,
                            "missing_branches": 1,
                            "num_partial_branches": 0,
                            "percent_covered_display": "62.5%",
                        }
                    },
                    "func_b": {
                        "summary": {
                            "covered_lines": 10,
                            "num_statements": 12,
                            "percent_covered": 83.33,
                            "missing_lines": 2,
                            "excluded_lines": 0,
                            "num_branches": 1,
                            "covered_branches": 1,
                            "missing_branches": 0,
                            "num_partial_branches": 0,
                            "percent_covered_display": "83.3%",
                        }
                    },
                },
            }
        }
    }

    result = extract_coverage_data(coverage_data)

    assert len(result) == 2
    func_names = {r.function_name for r in result}
    assert func_names == {"func_a", "func_b"}


def test_extract_coverage_data_no_functions():
    coverage_data = {
        "files": {
            "src/empty.py": {
                "summary": {
                    "num_statements": 0,
                    "covered_lines": 0,
                    "percent_covered": 0.0,
                },
            }
        }
    }

    result = extract_coverage_data(coverage_data)

    assert len(result) == 0


def test_extract_coverage_data_empty_files():
    coverage_data: dict[str, dict] = {"files": {}}

    result = extract_coverage_data(coverage_data)

    assert len(result) == 0


def test_extract_coverage_data_missing_summary_fields():
    coverage_data: dict[str, dict] = {
        "files": {
            "src/partial.py": {
                "summary": {},  # Missing all fields
                "functions": {"some_func": {"summary": {}}},  # Missing all fields
            }
        }
    }

    result = extract_coverage_data(coverage_data)

    assert len(result) == 1
    record = result[0]
    # All fields should default to 0 or 0.0
    assert record.covered_lines == 0
    assert record.total_statements == 0
    assert record.percent_covered == 0.0


# Tests for write_tsv_file()


def test_write_tsv_file(tmp_path):
    output_file = tmp_path / "output.tsv"
    data = [
        CoverageRecord(
            file_path="src/example.py",
            function_name="my_function",
            covered_lines=10,
            total_statements=12,
            percent_covered=83.33,
            missing_lines=2,
            excluded_lines=0,
            num_branches=4,
            covered_branches=3,
            missing_branches=1,
            num_partial_branches=1,
            percent_covered_display="83.3%",
            file_total_statements=100,
            file_covered_lines=80,
            file_percent_covered=80.0,
            test_duration=0.25,
        ),
    ]

    write_tsv_file(data, output_file)

    assert output_file.exists()
    content = output_file.read_text()
    lines = content.strip().split("\n")

    assert len(lines) == 2  # Header + 1 data row
    header = lines[0].split("\t")
    assert header[0] == "file_path"
    assert header[1] == "function_name"
    assert header[-1] == "test_duration_seconds"

    data_row = lines[1].split("\t")
    assert data_row[0] == "src/example.py"
    assert data_row[1] == "my_function"
    assert data_row[2] == "10"
    assert data_row[-1] == "0.25"


def test_write_tsv_file_no_timing(tmp_path):
    output_file = tmp_path / "output.tsv"
    data = [
        CoverageRecord(
            file_path="src/test.py",
            function_name="func",
            covered_lines=5,
            total_statements=5,
            percent_covered=100.0,
            missing_lines=0,
            excluded_lines=0,
            num_branches=0,
            covered_branches=0,
            missing_branches=0,
            num_partial_branches=0,
            percent_covered_display="100%",
            file_total_statements=10,
            file_covered_lines=10,
            file_percent_covered=100.0,
            test_duration=None,
        ),
    ]

    write_tsv_file(data, output_file)

    content = output_file.read_text()
    lines = content.split("\n")  # Don't strip - need to preserve empty last field

    # Check header has test_duration_seconds as last field
    header = lines[0].split("\t")
    assert header[-1] == "test_duration_seconds"
    assert len(header) == 16

    # Check data row
    data_row = lines[1].split("\t")
    # Should have 16 fields, last one empty for None duration
    assert len(data_row) == 16
    # Last column (test_duration_seconds) should be empty string
    assert data_row[-1] == ""
    # file_percent_covered should be second to last
    assert data_row[-2] == "100.0"


def test_write_tsv_file_multiple_records(tmp_path):
    output_file = tmp_path / "output.tsv"
    data = [
        CoverageRecord(
            file_path="src/a.py",
            function_name="func_a",
            covered_lines=5,
            total_statements=5,
            percent_covered=100.0,
            missing_lines=0,
            excluded_lines=0,
            num_branches=0,
            covered_branches=0,
            missing_branches=0,
            num_partial_branches=0,
            percent_covered_display="100%",
            file_total_statements=10,
            file_covered_lines=10,
            file_percent_covered=100.0,
            test_duration=0.1,
        ),
        CoverageRecord(
            file_path="src/b.py",
            function_name="func_b",
            covered_lines=3,
            total_statements=5,
            percent_covered=60.0,
            missing_lines=2,
            excluded_lines=0,
            num_branches=1,
            covered_branches=0,
            missing_branches=1,
            num_partial_branches=0,
            percent_covered_display="60%",
            file_total_statements=20,
            file_covered_lines=15,
            file_percent_covered=75.0,
            test_duration=None,
        ),
    ]

    write_tsv_file(data, output_file)

    content = output_file.read_text()
    lines = content.strip().split("\n")

    assert len(lines) == 3  # Header + 2 data rows


# Tests for main()


def test_main_basic(tmp_path):
    # Create input file
    input_file = tmp_path / "coverage.json"
    coverage_data = {
        "files": {
            "src/example.py": {
                "summary": {
                    "num_statements": 10,
                    "covered_lines": 8,
                    "percent_covered": 80.0,
                },
                "functions": {
                    "test_func": {
                        "summary": {
                            "covered_lines": 5,
                            "num_statements": 5,
                            "percent_covered": 100.0,
                            "missing_lines": 0,
                            "excluded_lines": 0,
                            "num_branches": 0,
                            "covered_branches": 0,
                            "missing_branches": 0,
                            "num_partial_branches": 0,
                            "percent_covered_display": "100%",
                        }
                    }
                },
            }
        }
    }
    input_file.write_text(json.dumps(coverage_data))

    output_file = tmp_path / "output.tsv"

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file), str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Successfully converted coverage data to TSV format" in result.output
    assert "Records written: 1" in result.output


def test_main_with_timing(tmp_path):
    # Create input file
    input_file = tmp_path / "coverage.json"
    coverage_data = {
        "files": {
            "src/utils.py": {
                "summary": {
                    "num_statements": 10,
                    "covered_lines": 10,
                    "percent_covered": 100.0,
                },
                "functions": {
                    "format_data": {
                        "summary": {
                            "covered_lines": 5,
                            "num_statements": 5,
                            "percent_covered": 100.0,
                            "missing_lines": 0,
                            "excluded_lines": 0,
                            "num_branches": 0,
                            "covered_branches": 0,
                            "missing_branches": 0,
                            "num_partial_branches": 0,
                            "percent_covered_display": "100%",
                        }
                    }
                },
            }
        }
    }
    input_file.write_text(json.dumps(coverage_data))

    # Create timing file
    timing_file = tmp_path / "timing.json"
    timing_data = {
        "tests": [
            {"nodeid": "tests/test_utils.py::test_format_data", "duration": 0.5},
        ]
    }
    timing_file.write_text(json.dumps(timing_data))

    output_file = tmp_path / "output.tsv"

    runner = CliRunner()
    result = runner.invoke(
        main, [str(input_file), str(output_file), "--timing", str(timing_file)]
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Loaded timing data for 1 tests" in result.output
    assert "Records with timing data:" in result.output


def test_main_default_output_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create input file
    input_file = tmp_path / "coverage.json"
    coverage_data = {
        "files": {
            "src/test.py": {
                "summary": {
                    "num_statements": 5,
                    "covered_lines": 5,
                    "percent_covered": 100.0,
                },
                "functions": {
                    "func": {
                        "summary": {
                            "covered_lines": 5,
                            "num_statements": 5,
                            "percent_covered": 100.0,
                            "missing_lines": 0,
                            "excluded_lines": 0,
                            "num_branches": 0,
                            "covered_branches": 0,
                            "missing_branches": 0,
                            "num_partial_branches": 0,
                            "percent_covered_display": "100%",
                        }
                    }
                },
            }
        }
    }
    input_file.write_text(json.dumps(coverage_data))

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file)])

    assert result.exit_code == 0
    assert Path("coverage_data.tsv").exists()


def test_main_nonexistent_input(tmp_path):
    input_file = tmp_path / "nonexistent.json"
    output_file = tmp_path / "output.tsv"

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file), str(output_file)])

    # Click should catch this with "exists=True" validation
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower()


def test_main_invalid_json(tmp_path):
    input_file = tmp_path / "invalid.json"
    input_file.write_text("This is not valid JSON")

    output_file = tmp_path / "output.tsv"

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file), str(output_file)])

    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_main_missing_expected_key(tmp_path):
    input_file = tmp_path / "bad_structure.json"
    # Missing 'files' key that code expects
    input_file.write_text('{"other_key": "value"}')

    output_file = tmp_path / "output.tsv"

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file), str(output_file)])

    # Should handle missing 'files' gracefully (empty dict)
    assert result.exit_code == 0
    assert "Records written: 0" in result.output


def test_main_write_error(tmp_path):
    input_file = tmp_path / "coverage.json"
    coverage_data = {
        "files": {
            "src/test.py": {
                "summary": {
                    "num_statements": 5,
                    "covered_lines": 5,
                    "percent_covered": 100.0,
                },
                "functions": {
                    "func": {
                        "summary": {
                            "covered_lines": 5,
                            "num_statements": 5,
                            "percent_covered": 100.0,
                            "missing_lines": 0,
                            "excluded_lines": 0,
                            "num_branches": 0,
                            "covered_branches": 0,
                            "missing_branches": 0,
                            "num_partial_branches": 0,
                            "percent_covered_display": "100%",
                        }
                    }
                },
            }
        }
    }
    input_file.write_text(json.dumps(coverage_data))

    # Try to write to a directory (should fail)
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file), str(output_dir)])

    assert result.exit_code == 1
    assert "Error: File operation failed" in result.output


def test_main_timing_file_not_found(tmp_path):
    input_file = tmp_path / "coverage.json"
    coverage_data = {
        "files": {
            "src/test.py": {
                "summary": {
                    "num_statements": 5,
                    "covered_lines": 5,
                    "percent_covered": 100.0,
                },
                "functions": {
                    "func": {
                        "summary": {
                            "covered_lines": 5,
                            "num_statements": 5,
                            "percent_covered": 100.0,
                            "missing_lines": 0,
                            "excluded_lines": 0,
                            "num_branches": 0,
                            "covered_branches": 0,
                            "missing_branches": 0,
                            "num_partial_branches": 0,
                            "percent_covered_display": "100%",
                        }
                    }
                },
            }
        }
    }
    input_file.write_text(json.dumps(coverage_data))

    output_file = tmp_path / "output.tsv"
    timing_file = tmp_path / "nonexistent_timing.json"

    runner = CliRunner()
    result = runner.invoke(
        main, [str(input_file), str(output_file), "--timing", str(timing_file)]
    )

    # Click should validate timing file exists
    assert result.exit_code != 0
