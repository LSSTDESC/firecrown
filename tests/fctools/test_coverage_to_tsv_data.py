"""Unit tests for coverage_to_tsv data processing functions.

Tests the data processing functions (match_test_to_function, extract_coverage_data,
and write_tsv_file).
"""

# pylint: disable=missing-function-docstring

import pytest

from firecrown.fctools.coverage_to_tsv import (
    CoverageRecord,
    extract_coverage_data,
    match_test_to_function,
    write_tsv_file,
)


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
