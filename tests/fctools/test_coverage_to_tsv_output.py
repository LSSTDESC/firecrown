"""Unit tests for output and CLI functions in firecrown.fctools.coverage_to_tsv module.

Tests the TSV file writing and main CLI function.
"""

# pylint: disable=missing-function-docstring

import json
import subprocess
import sys
from pathlib import Path

from firecrown.fctools.coverage_to_tsv import CoverageRecord, main, write_tsv_file

from . import match_wrapped

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

    script_path = "firecrown/fctools/coverage_to_tsv.py"
    result = subprocess.run(
        [sys.executable, script_path, str(input_file), str(output_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_file.exists()
    assert match_wrapped(
        result.stdout, "Successfully converted coverage data to TSV format"
    )
    assert match_wrapped(result.stdout, "Records written: 1")


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

    script_path = "firecrown/fctools/coverage_to_tsv.py"
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            str(input_file),
            str(output_file),
            "--timing",
            str(timing_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_file.exists()
    assert match_wrapped(result.stdout, "Loaded timing data for 1 tests")
    assert match_wrapped(result.stdout, "Records with timing data:")


def test_main_default_output_filename(tmp_path, monkeypatch):
    # Get absolute path to script before changing directory
    script_path = Path("firecrown/fctools/coverage_to_tsv.py").absolute()

    # Change to the tmp directory so the output file appears there
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

    result = subprocess.run(
        [sys.executable, str(script_path), str(input_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert Path("coverage_data.tsv").exists()


def test_main_nonexistent_input(tmp_path):
    input_file = tmp_path / "nonexistent.json"
    output_file = tmp_path / "output.tsv"

    script_path = "firecrown/fctools/coverage_to_tsv.py"
    result = subprocess.run(
        [sys.executable, script_path, str(input_file), str(output_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    # Typer should catch this with "exists=True" validation
    assert result.returncode != 0
    assert match_wrapped(result.stderr.lower(), "does not exist")


def test_main_invalid_json(tmp_path):
    input_file = tmp_path / "invalid.json"
    input_file.write_text("This is not valid JSON")

    output_file = tmp_path / "output.tsv"

    script_path = "firecrown/fctools/coverage_to_tsv.py"
    result = subprocess.run(
        [sys.executable, script_path, str(input_file), str(output_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ERROR" in result.stderr


def test_main_missing_expected_key(tmp_path):
    input_file = tmp_path / "bad_structure.json"
    # Missing 'files' key that code expects
    input_file.write_text('{"other_key": "value"}')

    output_file = tmp_path / "output.tsv"

    script_path = "firecrown/fctools/coverage_to_tsv.py"
    result = subprocess.run(
        [sys.executable, script_path, str(input_file), str(output_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should handle missing 'files' gracefully (empty dict)
    assert result.returncode == 0
    assert match_wrapped(result.stdout, "Records written: 0")


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

    script_path = "firecrown/fctools/coverage_to_tsv.py"
    result = subprocess.run(
        [sys.executable, script_path, str(input_file), str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    # Error message appears in stdout, not stderr
    assert match_wrapped(result.stdout, "Error: File operation failed")


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

    script_path = "firecrown/fctools/coverage_to_tsv.py"
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            str(input_file),
            str(output_file),
            "--timing",
            str(timing_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Typer should validate timing file exists
    assert result.returncode != 0


# Tests for main() function called directly (for coverage of console output)


def test_main_direct_basic(tmp_path, capsys):
    """Test main function called directly for coverage of success path console output.

    This test calls main() directly (not via subprocess) to ensure coverage
    tracking for the console.print statements in the success path (lines 338-370).
    """
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

    # Call main directly; None is acceptable for Typer optional params
    main(
        input_file=input_file,
        output_file=output_file,
        timing=None,  # type: ignore[arg-type]
    )

    # Check output file was created
    assert output_file.exists()

    # Verify console output (Rich may wrap long paths with newlines)
    captured = capsys.readouterr()
    assert "Reading coverage data from" in captured.out
    assert "coverage.json" in captured.out
    assert "Extracting function-level coverage data" in captured.out
    assert "Writing 1 records to" in captured.out
    assert "Successfully converted coverage data to TSV format" in captured.out
    assert "Output file:" in captured.out
    assert "Records written: 1" in captured.out


def test_main_direct_with_timing(tmp_path, capsys):
    """Test main function with timing data for coverage of timing-related output.

    Tests the branch where timing_data is not None (lines 367-370).
    """
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

    # Create timing file
    timing_file = tmp_path / "timing.txt"
    timing_file.write_text("test_func 0.123s call\n")

    output_file = tmp_path / "output.tsv"

    # Call main directly with timing (use keyword args for mypy)
    main(input_file=input_file, output_file=output_file, timing=timing_file)

    # Check output file was created
    assert output_file.exists()

    # Verify console output includes timing-related messages (Rich may wrap paths)
    captured = capsys.readouterr()
    assert match_wrapped(captured.out, "Reading timing data from")
    assert match_wrapped(captured.out, "timing.txt")
    assert match_wrapped(captured.out, "Loaded timing data for 1 tests")
    assert match_wrapped(captured.out, "Records with timing data:")
    assert match_wrapped(
        captured.out, "Successfully converted coverage data to TSV format"
    )
