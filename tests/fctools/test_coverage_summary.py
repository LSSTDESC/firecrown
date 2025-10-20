"""Unit tests for firecrown.fctools.coverage_summary module.

Tests the coverage analysis and reporting functionality.
"""

import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from firecrown.fctools.coverage_summary import (
    CoverageSummary,
    FileIssue,
    _analyze_single_file,
    _calculate_branch_coverage_summary,
    _calculate_file_branch_coverage,
    _create_file_issue,
    _print_file_issue_details,
    _print_source_code_for_missing_lines,
    analyze_coverage_json,
    app,
    print_coverage_summary,
    print_file_issues,
    print_perfect_coverage_files,
)

from . import match_wrapped


@pytest.fixture
def console():
    """Return a rich console object for testing."""
    return Console()


class TestCoverageSummary:
    """Tests for CoverageSummary dataclass."""

    def test_default_values(self):
        """Test that CoverageSummary has correct default values."""
        summary = CoverageSummary()

        assert summary.total_files == 0
        assert summary.files_with_perfect_coverage == 0
        assert summary.files_with_missing_lines == 0
        assert summary.files_with_missing_branches == 0
        assert summary.files_with_excluded_lines == 0
        assert summary.overall_line_coverage == 0.0
        assert summary.overall_branch_coverage == 0.0
        assert summary.total_statements == 0
        assert summary.total_missing_lines == 0
        assert summary.total_excluded_lines == 0
        assert summary.total_branches == 0
        assert summary.total_missing_branches == 0

    def test_custom_values(self):
        """Test creating CoverageSummary with custom values."""
        summary = CoverageSummary(
            total_files=10,
            files_with_perfect_coverage=5,
            overall_line_coverage=85.5,
        )

        assert summary.total_files == 10
        assert summary.files_with_perfect_coverage == 5
        assert summary.overall_line_coverage == 85.5


class TestFileIssue:
    """Tests for FileIssue dataclass."""

    def test_file_issue_creation(self):
        """Test creating a FileIssue object."""
        issue = FileIssue(
            file_path="test.py",
            line_coverage=85.5,
            branch_coverage=75.0,
            missing_lines_count=5,
            missing_branches_count=2,
            excluded_lines_count=3,
            missing_lines=[10, 20, 30],
            missing_branches=[[15, 16], [25, 26]],
            excluded_lines=[5, 6, 7],
            total_statements=100,
            total_branches=10,
        )

        assert issue.file_path == "test.py"
        assert issue.line_coverage == 85.5
        assert issue.branch_coverage == 75.0
        assert issue.missing_lines == [10, 20, 30]
        assert issue.missing_branches == [[15, 16], [25, 26]]
        assert issue.excluded_lines == [5, 6, 7]


class TestCalculateBranchCoverageSummary:
    """Tests for _calculate_branch_coverage_summary function."""

    def test_with_branches(self):
        """Test branch coverage calculation with branches present."""
        totals = {"num_branches": 100, "missing_branches": 20}

        total, missing, coverage = _calculate_branch_coverage_summary(totals)

        assert total == 100
        assert missing == 20
        assert coverage == 80.0

    def test_with_no_branches(self):
        """Test branch coverage when no branches exist."""
        totals = {"num_branches": 0, "missing_branches": 0}

        total, missing, coverage = _calculate_branch_coverage_summary(totals)

        assert total == 0
        assert missing == 0
        assert coverage == 100.0

    def test_missing_keys(self):
        """Test with missing dictionary keys."""
        totals: dict[str, int] = {}

        total, missing, coverage = _calculate_branch_coverage_summary(totals)

        assert total == 0
        assert missing == 0
        assert coverage == 100.0

    def test_perfect_branch_coverage(self):
        """Test with 100% branch coverage."""
        totals = {"num_branches": 50, "missing_branches": 0}

        total, missing, coverage = _calculate_branch_coverage_summary(totals)

        assert total == 50
        assert missing == 0
        assert coverage == 100.0


class TestCalculateFileBranchCoverage:
    """Tests for _calculate_file_branch_coverage function."""

    def test_with_branches(self):
        """Test file branch coverage with branches present."""
        file_summary = {"num_branches": 10, "covered_branches": 8}

        coverage = _calculate_file_branch_coverage(file_summary, total_statements=50)

        assert coverage == 80.0

    def test_no_branches_with_statements(self):
        """Test file with no branches but has statements."""
        file_summary = {"num_branches": 0}

        coverage = _calculate_file_branch_coverage(file_summary, total_statements=50)

        assert coverage == 100.0

    def test_no_branches_no_statements(self):
        """Test file with no branches and no statements."""
        file_summary = {"num_branches": 0}

        coverage = _calculate_file_branch_coverage(file_summary, total_statements=0)

        assert coverage == 0.0

    def test_perfect_branch_coverage(self):
        """Test file with 100% branch coverage."""
        file_summary = {"num_branches": 20, "covered_branches": 20}

        coverage = _calculate_file_branch_coverage(file_summary, total_statements=100)

        assert coverage == 100.0

    def test_missing_keys(self):
        """Test with missing dictionary keys."""
        file_summary: dict[str, int] = {}

        coverage = _calculate_file_branch_coverage(file_summary, total_statements=50)

        assert coverage == 100.0


class TestCreateFileIssue:
    """Tests for _create_file_issue function."""

    def test_create_file_issue(self):
        """Test creating a FileIssue from coverage data."""
        file_summary = {"num_statements": 100, "num_branches": 20}
        missing_lines = [10, 20, 30]
        missing_branches = [[15, 16]]
        excluded_lines = [5, 6]

        issue = _create_file_issue(
            "test.py",
            file_summary,
            missing_lines,
            missing_branches,
            excluded_lines,
            85.0,
            90.0,
        )

        assert issue.file_path == "test.py"
        assert issue.line_coverage == 85.0
        assert issue.branch_coverage == 90.0
        assert issue.missing_lines_count == 3
        assert issue.missing_branches_count == 1
        assert issue.excluded_lines_count == 2
        assert issue.missing_lines == missing_lines
        assert issue.missing_branches == missing_branches
        assert issue.excluded_lines == excluded_lines
        assert issue.total_statements == 100
        assert issue.total_branches == 20


class TestAnalyzeSingleFile:
    """Tests for _analyze_single_file function."""

    def test_perfect_coverage_file(self):
        """Test analyzing a file with perfect coverage."""
        summary = CoverageSummary()
        file_data = {
            "summary": {
                "num_statements": 100,
                "percent_covered": 100.0,
                "num_branches": 10,
                "covered_branches": 10,
            },
            "missing_lines": [],
            "missing_branches": [],
            "excluded_lines": [],
        }

        result = _analyze_single_file("perfect.py", file_data, summary)

        assert result is None
        assert summary.total_files == 1
        assert summary.files_with_perfect_coverage == 1
        assert summary.files_with_missing_lines == 0
        assert summary.files_with_missing_branches == 0
        assert summary.files_with_excluded_lines == 0

    def test_file_with_missing_lines(self):
        """Test analyzing a file with missing lines."""
        summary = CoverageSummary()
        file_data = {
            "summary": {
                "num_statements": 100,
                "percent_covered": 90.0,
                "num_branches": 0,
            },
            "missing_lines": [10, 20, 30],
            "missing_branches": [],
            "excluded_lines": [],
        }

        result = _analyze_single_file("incomplete.py", file_data, summary)

        assert result is not None
        assert result.file_path == "incomplete.py"
        assert result.missing_lines_count == 3
        assert summary.total_files == 1
        assert summary.files_with_perfect_coverage == 0
        assert summary.files_with_missing_lines == 1

    def test_file_with_missing_branches(self):
        """Test analyzing a file with missing branches."""
        summary = CoverageSummary()
        file_data = {
            "summary": {
                "num_statements": 100,
                "percent_covered": 100.0,
                "num_branches": 10,
                "covered_branches": 8,
            },
            "missing_lines": [],
            "missing_branches": [[15, 16], [25, 26]],
            "excluded_lines": [],
        }

        result = _analyze_single_file("branches.py", file_data, summary)

        assert result is not None
        assert result.missing_branches_count == 2
        assert summary.files_with_missing_branches == 1

    def test_file_with_excluded_lines_only(self):
        """Test analyzing a file with only excluded lines (no missing coverage)."""
        summary = CoverageSummary()
        file_data = {
            "summary": {
                "num_statements": 100,
                "percent_covered": 100.0,
                "num_branches": 0,
            },
            "missing_lines": [],
            "missing_branches": [],
            "excluded_lines": [5, 6, 7],
        }

        result = _analyze_single_file("excluded.py", file_data, summary)

        assert result is not None
        assert result.excluded_lines_count == 3
        assert summary.files_with_excluded_lines == 1
        assert summary.files_with_perfect_coverage == 1  # Still perfect coverage

    def test_file_with_all_issues(self):
        """Test file with missing lines, branches, and excluded lines."""
        summary = CoverageSummary()
        file_data = {
            "summary": {
                "num_statements": 100,
                "percent_covered": 80.0,
                "num_branches": 20,
                "covered_branches": 15,
            },
            "missing_lines": [10, 20],
            "missing_branches": [[30, 31]],
            "excluded_lines": [40, 41],
        }

        result = _analyze_single_file("issues.py", file_data, summary)

        assert result is not None
        assert summary.files_with_missing_lines == 1
        assert summary.files_with_missing_branches == 1
        assert summary.files_with_excluded_lines == 1
        assert summary.files_with_perfect_coverage == 0


class TestAnalyzeCoverageJson:
    """Tests for analyze_coverage_json function."""

    def test_analyze_simple_coverage(self, tmp_path, console):
        """Test analyzing a simple coverage JSON file."""
        coverage_data = {
            "files": {
                "file1.py": {
                    "summary": {
                        "num_statements": 100,
                        "percent_covered": 90.0,
                        "num_branches": 10,
                        "covered_branches": 9,
                    },
                    "missing_lines": [50],
                    "missing_branches": [[60, 61]],
                    "excluded_lines": [],
                },
                "file2.py": {
                    "summary": {
                        "num_statements": 50,
                        "percent_covered": 100.0,
                        "num_branches": 0,
                    },
                    "missing_lines": [],
                    "missing_branches": [],
                    "excluded_lines": [],
                },
            },
            "totals": {
                "num_statements": 150,
                "missing_lines": 1,
                "excluded_lines": 0,
                "percent_covered": 93.3,
                "num_branches": 10,
                "missing_branches": 1,
            },
        }

        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        summary, file_issues = analyze_coverage_json(console, coverage_file)

        assert summary.total_files == 2
        assert summary.files_with_perfect_coverage == 1
        assert summary.total_statements == 150
        assert summary.overall_line_coverage == 93.3
        assert len(file_issues) == 1
        assert file_issues[0].file_path == "file1.py"

    def test_file_issues_sorted_by_coverage(self, tmp_path, console):
        """Test that file issues are sorted by coverage (worst first)."""
        coverage_data = {
            "files": {
                "file1.py": {
                    "summary": {
                        "num_statements": 100,
                        "percent_covered": 90.0,
                        "num_branches": 0,
                    },
                    "missing_lines": [1],
                    "missing_branches": [],
                    "excluded_lines": [],
                },
                "file2.py": {
                    "summary": {
                        "num_statements": 100,
                        "percent_covered": 50.0,
                        "num_branches": 0,
                    },
                    "missing_lines": [1, 2, 3],
                    "missing_branches": [],
                    "excluded_lines": [],
                },
                "file3.py": {
                    "summary": {
                        "num_statements": 100,
                        "percent_covered": 75.0,
                        "num_branches": 0,
                    },
                    "missing_lines": [1, 2],
                    "missing_branches": [],
                    "excluded_lines": [],
                },
            },
            "totals": {
                "num_statements": 300,
                "missing_lines": 6,
                "excluded_lines": 0,
                "percent_covered": 71.7,
                "num_branches": 0,
                "missing_branches": 0,
            },
        }

        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        _, file_issues = analyze_coverage_json(console, coverage_file)

        assert len(file_issues) == 3
        assert file_issues[0].file_path == "file2.py"  # Worst: 50%
        assert file_issues[1].file_path == "file3.py"  # Middle: 75%
        assert file_issues[2].file_path == "file1.py"  # Best: 90%

    def test_empty_coverage_file(self, tmp_path, console):
        """Test analyzing an empty coverage file."""
        coverage_data: dict[str, dict] = {"files": {}, "totals": {}}

        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        summary, file_issues = analyze_coverage_json(console, coverage_file)

        assert summary.total_files == 0
        assert len(file_issues) == 0


class TestPrintCoverageSummary:
    """Tests for print_coverage_summary function."""

    def test_print_summary(self, capsys, console):
        """Test printing coverage summary."""
        summary = CoverageSummary(
            total_files=10,
            files_with_perfect_coverage=5,
            files_with_missing_lines=3,
            files_with_missing_branches=2,
            files_with_excluded_lines=1,
            overall_line_coverage=85.5,
            overall_branch_coverage=90.0,
            total_statements=1000,
            total_missing_lines=145,
            total_excluded_lines=10,
            total_branches=100,
            total_missing_branches=10,
        )

        print_coverage_summary(console, summary)

        captured = capsys.readouterr()
        assert "COVERAGE ANALYSIS SUMMARY" in captured.out
        assert "Total files analyzed: 10" in captured.out
        assert "Files with perfect coverage: 5" in captured.out
        assert "Overall line coverage: 85.5%" in captured.out
        assert "Overall branch coverage: 90.0%" in captured.out
        assert "Total statements: 1000" in captured.out
        assert "Missing lines: 145" in captured.out


class TestPrintSourceCodeForMissingLines:
    """Tests for _print_source_code_for_missing_lines function."""

    def test_print_source_existing_file(self, tmp_path, capsys, console):
        """Test printing source code for existing file."""
        source_file = tmp_path / "test.py"
        source_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        issue = FileIssue(
            file_path=str(source_file),
            line_coverage=80.0,
            branch_coverage=100.0,
            missing_lines_count=2,
            missing_branches_count=0,
            excluded_lines_count=0,
            missing_lines=[2, 4],
            missing_branches=[],
            excluded_lines=[],
            total_statements=5,
            total_branches=0,
        )

        _print_source_code_for_missing_lines(console, issue)

        captured = capsys.readouterr()
        assert "Source code for missing lines:" in captured.out
        assert "2: line 2" in captured.out
        assert "4: line 4" in captured.out

    def test_print_source_nonexistent_file(self, capsys, console):
        """Test handling of nonexistent source file."""
        issue = FileIssue(
            file_path="/nonexistent/file.py",
            line_coverage=80.0,
            branch_coverage=100.0,
            missing_lines_count=1,
            missing_branches_count=0,
            excluded_lines_count=0,
            missing_lines=[10],
            missing_branches=[],
            excluded_lines=[],
            total_statements=100,
            total_branches=0,
        )

        _print_source_code_for_missing_lines(console, issue)

        captured = capsys.readouterr()
        assert "(Source file not found for line details)" in captured.out

    def test_print_source_with_unicode_error(self, tmp_path, capsys, console):
        """Test handling of files with encoding errors."""
        source_file = tmp_path / "test.py"
        # Write binary data that's not valid UTF-8
        source_file.write_bytes(b"\xff\xfe\x00\x00")

        issue = FileIssue(
            file_path=str(source_file),
            line_coverage=80.0,
            branch_coverage=100.0,
            missing_lines_count=1,
            missing_branches_count=0,
            excluded_lines_count=0,
            missing_lines=[1],
            missing_branches=[],
            excluded_lines=[],
            total_statements=10,
            total_branches=0,
        )

        _print_source_code_for_missing_lines(console, issue)

        captured = capsys.readouterr()
        assert "(Error reading source file:" in captured.out

    def test_print_source_line_out_of_range(self, tmp_path, capsys, console):
        """Test handling of line numbers outside file range."""
        source_file = tmp_path / "test.py"
        source_file.write_text("line 1\nline 2\n")

        issue = FileIssue(
            file_path=str(source_file),
            line_coverage=50.0,
            branch_coverage=100.0,
            missing_lines_count=2,
            missing_branches_count=0,
            excluded_lines_count=0,
            missing_lines=[1, 100],  # Line 100 doesn't exist
            missing_branches=[],
            excluded_lines=[],
            total_statements=4,
            total_branches=0,
        )

        _print_source_code_for_missing_lines(console, issue)

        captured = capsys.readouterr()
        assert "1: line 1" in captured.out
        assert "100:" not in captured.out  # Should not print out-of-range lines


class TestPrintFileIssueDetails:
    """Tests for _print_file_issue_details function."""

    def test_print_with_missing_lines_and_branches(self, tmp_path, capsys, console):
        """Test printing file issue with missing lines and branches."""
        source_file = tmp_path / "test.py"
        source_file.write_text("line 1\nline 2\nline 3\n")

        issue = FileIssue(
            file_path=str(source_file),
            line_coverage=80.0,
            branch_coverage=75.0,
            missing_lines_count=2,
            missing_branches_count=1,
            excluded_lines_count=0,
            missing_lines=[2, 3],
            missing_branches=[[10, 11]],
            excluded_lines=[],
            total_statements=10,
            total_branches=4,
        )

        _print_file_issue_details(console, issue, show_source=True)

        captured = capsys.readouterr()
        assert "Line Coverage: 80.0%" in captured.out
        assert "(8/10 statements)" in captured.out
        assert "Branch Coverage: 75.0%" in captured.out
        assert "(3/4 branches)" in captured.out
        assert "Missing Lines (2):" in captured.out
        assert "Missing Branches (1):" in captured.out
        assert "[10, 11]" in captured.out

    def test_print_without_showing_source(self, capsys, console):
        """Test printing without showing source code."""
        issue = FileIssue(
            file_path="test.py",
            line_coverage=90.0,
            branch_coverage=100.0,
            missing_lines_count=1,
            missing_branches_count=0,
            excluded_lines_count=0,
            missing_lines=[50],
            missing_branches=[],
            excluded_lines=[],
            total_statements=100,
            total_branches=0,
        )

        _print_file_issue_details(console, issue, show_source=False)

        captured = capsys.readouterr()
        assert "Line Coverage: 90.0%" in captured.out
        assert "Missing Lines (1):" in captured.out
        assert "Source code for missing lines:" not in captured.out

    def test_print_with_no_branches(self, capsys, console):
        """Test printing for file with no branches."""
        issue = FileIssue(
            file_path="test.py",
            line_coverage=100.0,
            branch_coverage=100.0,
            missing_lines_count=0,
            missing_branches_count=0,
            excluded_lines_count=1,
            missing_lines=[],
            missing_branches=[],
            excluded_lines=[10],
            total_statements=50,
            total_branches=0,
        )

        _print_file_issue_details(console, issue, show_source=False)

        captured = capsys.readouterr()
        assert "Line Coverage: 100.0%" in captured.out
        assert "Branch Coverage:" not in captured.out  # Should not show for 0 branches
        assert "Excluded Lines (1):" in captured.out

    def test_print_with_excluded_lines(self, capsys, console):
        """Test printing with excluded lines."""
        issue = FileIssue(
            file_path="test.py",
            line_coverage=100.0,
            branch_coverage=100.0,
            missing_lines_count=0,
            missing_branches_count=0,
            excluded_lines_count=3,
            missing_lines=[],
            missing_branches=[],
            excluded_lines=[5, 6, 7],
            total_statements=50,
            total_branches=0,
        )

        _print_file_issue_details(console, issue, show_source=False)

        captured = capsys.readouterr()
        assert "Excluded Lines (3):" in captured.out
        assert "5-7" in captured.out


class TestPrintFileIssues:
    """Tests for print_file_issues function."""

    def test_print_no_issues(self, capsys, console):
        """Test printing when no issues exist."""
        print_file_issues(console, [])

        captured = capsys.readouterr()
        assert "ALL FILES HAVE PERFECT COVERAGE!" in captured.out

    def test_print_multiple_issues(self, capsys, console):
        """Test printing multiple file issues."""
        issues = [
            FileIssue(
                file_path="file1.py",
                line_coverage=80.0,
                branch_coverage=100.0,
                missing_lines_count=2,
                missing_branches_count=0,
                excluded_lines_count=0,
                missing_lines=[10, 20],
                missing_branches=[],
                excluded_lines=[],
                total_statements=10,
                total_branches=0,
            ),
            FileIssue(
                file_path="file2.py",
                line_coverage=90.0,
                branch_coverage=85.0,
                missing_lines_count=1,
                missing_branches_count=1,
                excluded_lines_count=0,
                missing_lines=[15],
                missing_branches=[[25, 26]],
                excluded_lines=[],
                total_statements=10,
                total_branches=5,
            ),
        ]

        print_file_issues(console, issues, show_source=False)

        captured = capsys.readouterr()
        assert "FILES WITH COVERAGE ISSUES OR EXCLUDED LINES:" in captured.out
        assert "1. file1.py" in captured.out
        assert "2. file2.py" in captured.out


class TestPrintPerfectCoverageFiles:
    """Tests for print_perfect_coverage_files function."""

    def test_print_perfect_files(self, capsys, console):
        """Test printing files with perfect coverage."""
        file_issues = [
            FileIssue(
                file_path="bad.py",
                line_coverage=80.0,
                branch_coverage=100.0,
                missing_lines_count=1,
                missing_branches_count=0,
                excluded_lines_count=0,
                missing_lines=[10],
                missing_branches=[],
                excluded_lines=[],
                total_statements=10,
                total_branches=0,
            ),
        ]
        all_files = ["bad.py", "good1.py", "good2.py"]

        print_perfect_coverage_files(console, file_issues, all_files)

        captured = capsys.readouterr()
        assert "FILES WITH PERFECT COVERAGE:" in captured.out
        assert "✅ good1.py" in captured.out
        assert "✅ good2.py" in captured.out
        assert "bad.py" not in captured.out

    def test_no_perfect_files(self, capsys, console):
        """Test when no files have perfect coverage."""
        file_issues = [
            FileIssue(
                file_path="file1.py",
                line_coverage=80.0,
                branch_coverage=100.0,
                missing_lines_count=1,
                missing_branches_count=0,
                excluded_lines_count=0,
                missing_lines=[10],
                missing_branches=[],
                excluded_lines=[],
                total_statements=10,
                total_branches=0,
            ),
        ]
        all_files = ["file1.py"]

        print_perfect_coverage_files(console, file_issues, all_files)

        captured = capsys.readouterr()
        # Should not print anything if no perfect files
        assert captured.out == ""


class TestMainFunction:  # pylint: disable=import-outside-toplevel
    """Tests for main CLI function."""

    def test_main_basic_usage(self, tmp_path):
        """Test main function with basic coverage file."""
        coverage_data = {
            "files": {
                "test.py": {
                    "summary": {
                        "num_statements": 100,
                        "percent_covered": 90.0,
                        "num_branches": 0,
                    },
                    "missing_lines": [50],
                    "missing_branches": [],
                    "excluded_lines": [],
                },
            },
            "totals": {
                "num_statements": 100,
                "missing_lines": 10,
                "excluded_lines": 0,
                "percent_covered": 90.0,
                "num_branches": 0,
                "missing_branches": 0,
            },
        }

        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        runner = CliRunner()
        result = runner.invoke(app, [str(coverage_file)])

        assert result.exit_code == 0

        assert match_wrapped(result.stdout, "COVERAGE ANALYSIS SUMMARY")
        assert match_wrapped(result.stdout, "Total files analyzed: 1")

    def test_main_with_show_source(self, tmp_path):
        """Test main function with --show-source flag."""
        source_file = tmp_path / "test.py"
        source_file.write_text("line 1\nline 2\n")

        coverage_data = {
            "files": {
                str(source_file): {
                    "summary": {
                        "num_statements": 2,
                        "percent_covered": 50.0,
                        "num_branches": 0,
                    },
                    "missing_lines": [2],
                    "missing_branches": [],
                    "excluded_lines": [],
                },
            },
            "totals": {
                "num_statements": 2,
                "missing_lines": 1,
                "excluded_lines": 0,
                "percent_covered": 50.0,
                "num_branches": 0,
                "missing_branches": 0,
            },
        }

        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        runner = CliRunner()
        result = runner.invoke(app, [str(coverage_file), "--show-source"])

        assert result.exit_code == 0

        assert match_wrapped(result.stdout, "Source code for missing lines:")

    def test_main_with_show_perfect(self, tmp_path):
        """Test main function with --show-perfect flag."""
        coverage_data = {
            "files": {
                "perfect.py": {
                    "summary": {
                        "num_statements": 100,
                        "percent_covered": 100.0,
                        "num_branches": 0,
                    },
                    "missing_lines": [],
                    "missing_branches": [],
                    "excluded_lines": [],
                },
                "imperfect.py": {
                    "summary": {
                        "num_statements": 100,
                        "percent_covered": 90.0,
                        "num_branches": 0,
                    },
                    "missing_lines": [50],
                    "missing_branches": [],
                    "excluded_lines": [],
                },
            },
            "totals": {
                "num_statements": 200,
                "missing_lines": 10,
                "excluded_lines": 0,
                "percent_covered": 95.0,
                "num_branches": 0,
                "missing_branches": 0,
            },
        }

        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        runner = CliRunner()
        result = runner.invoke(app, [str(coverage_file), "--show-perfect"])

        assert result.exit_code == 0

        assert match_wrapped(result.stdout, "FILES WITH PERFECT COVERAGE:")
        assert match_wrapped(result.stdout, "perfect.py")

    def test_main_with_nonexistent_file(self, tmp_path):
        """Test main function with nonexistent coverage file."""
        runner = CliRunner()
        result = runner.invoke(app, [str(tmp_path / "nonexistent.json")])

        assert result.exit_code != 0

    def test_main_with_os_error(self, tmp_path, monkeypatch):
        """Test main function handling of OSError."""
        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text('{"files": {}, "totals": {}}')

        # Mock load_json_file to raise OSError
        def mock_load_json_file(*args, **kwargs):
            raise OSError("Simulated OS error")

        from firecrown.fctools import coverage_summary

        monkeypatch.setattr(coverage_summary, "load_json_file", mock_load_json_file)

        runner = CliRunner()
        result = runner.invoke(app, [str(coverage_file)])

        assert result.exit_code == 1

        assert match_wrapped(result.stdout, "Error analyzing coverage file:")

    def test_main_block_with_subprocess(self, tmp_path):
        """Test that the script can be executed directly via subprocess.

        This test verifies that the __main__ block and ImportError fallback
        work correctly when the script is run directly. These lines cannot
        be covered by normal pytest-cov because they only execute in
        standalone script mode.
        """
        import subprocess
        import sys

        # Create a valid coverage file
        coverage_data = {
            "files": {},
            "totals": {
                "num_statements": 0,
                "missing_lines": 0,
                "excluded_lines": 0,
                "percent_covered": 100.0,
                "num_branches": 0,
                "missing_branches": 0,
            },
        }

        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        # Execute the script directly to test __main__ block and ImportError fallback
        script_path = "firecrown/fctools/coverage_summary.py"
        result = subprocess.run(
            [sys.executable, script_path, str(coverage_file)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0

        assert match_wrapped(result.stdout, "COVERAGE ANALYSIS SUMMARY")
        assert match_wrapped(result.stdout, "Overall Statistics:")
