"""Unit tests for coverage_to_tsv CLI integration.

Tests the main() function CLI interface using subprocess.
"""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path
import subprocess
import sys


def test_main_basic(tmp_path):
    coverage_file = tmp_path / "coverage.json"
    output_file = tmp_path / "output.tsv"

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

    coverage_file.write_text(json.dumps(coverage_data))

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/coverage_to_tsv.py",
            str(coverage_file),
            str(output_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_file.exists()

    content = output_file.read_text()
    lines = content.strip().split("\n")

    assert len(lines) == 2  # Header + 1 data row
    assert "my_function" in lines[1]
    assert "src/example.py" in lines[1]


def test_main_with_timing(tmp_path):
    coverage_file = tmp_path / "coverage.json"
    timing_file = tmp_path / "timing.json"
    output_file = tmp_path / "output.tsv"

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

    timing_data = {
        "tests": [
            {"nodeid": "tests/test_utils.py::test_format_data", "duration": 0.5},
        ]
    }

    coverage_file.write_text(json.dumps(coverage_data))
    timing_file.write_text(json.dumps(timing_data))

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/coverage_to_tsv.py",
            str(coverage_file),
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
    assert "Loaded timing data for 1 tests" in result.stdout
    assert "Records with timing data:" in result.stdout


def test_main_default_output_filename(tmp_path):
    coverage_file = tmp_path / "coverage.json"

    coverage_data = {
        "files": {
            "src/test.py": {
                "summary": {
                    "num_statements": 10,
                    "covered_lines": 10,
                    "percent_covered": 100.0,
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

    coverage_file.write_text(json.dumps(coverage_data))

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/coverage_to_tsv.py",
            str(coverage_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    # Default output file should be created in current directory
    default_output = Path("coverage_data.tsv")
    assert default_output.exists()


def test_main_nonexistent_input(tmp_path):
    nonexistent_file = tmp_path / "does_not_exist.json"
    output_file = tmp_path / "output.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/coverage_to_tsv.py",
            str(nonexistent_file),
            str(output_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not exist" in result.stderr.lower()


def test_main_invalid_json(tmp_path):
    coverage_file = tmp_path / "invalid.json"
    output_file = tmp_path / "output.tsv"

    coverage_file.write_text("This is not valid JSON{{{")

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/coverage_to_tsv.py",
            str(coverage_file),
            str(output_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "json" in result.stderr.lower()


def test_main_missing_expected_key(tmp_path):
    coverage_file = tmp_path / "missing_key.json"
    output_file = tmp_path / "output.tsv"

    # Missing the 'files' key at top level
    coverage_data: dict[str, dict] = {"something_else": {}}

    coverage_file.write_text(json.dumps(coverage_data))

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/coverage_to_tsv.py",
            str(coverage_file),
            str(output_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should handle gracefully - creates output with 0 records
    assert result.returncode == 0
    assert output_file.exists()

    content = output_file.read_text()
    lines = content.strip().split("\n")

    # Just the header, no data
    assert len(lines) == 1


def test_main_write_error(tmp_path):
    coverage_file = tmp_path / "coverage.json"

    coverage_data = {
        "files": {
            "src/test.py": {
                "summary": {
                    "num_statements": 10,
                    "covered_lines": 10,
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

    coverage_file.write_text(json.dumps(coverage_data))

    # Try to write to a directory (should fail)
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/coverage_to_tsv.py",
            str(coverage_file),
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Error: File operation failed" in result.stdout


def test_main_timing_file_not_found(tmp_path):
    coverage_file = tmp_path / "coverage.json"
    output_file = tmp_path / "output.tsv"
    nonexistent_timing = tmp_path / "timing_does_not_exist.txt"

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

    coverage_file.write_text(json.dumps(coverage_data))

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/coverage_to_tsv.py",
            str(coverage_file),
            str(output_file),
            "--timing",
            str(nonexistent_timing),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Typer should validate timing file exists
    assert result.returncode != 0
