#!/usr/bin/env python
"""Comprehensive tool to analyze test coverage output in JSON format.

This tool provides a detailed analysis of test coverage including:
- Overall coverage summary
- Per-file coverage breakdown
- Detailed information about untested lines and branches
- Files with less than perfect coverage
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from .common import format_line_ranges, load_json_file
else:
    try:
        from .common import format_line_ranges, load_json_file
    except ImportError:  # pragma: no cover
        from common import format_line_ranges, load_json_file


@dataclass
class CoverageSummary:
    """Summary statistics for coverage analysis."""

    total_files: int = 0
    files_with_perfect_coverage: int = 0
    files_with_missing_lines: int = 0
    files_with_missing_branches: int = 0
    files_with_excluded_lines: int = 0
    overall_line_coverage: float = 0.0
    overall_branch_coverage: float = 0.0
    total_statements: int = 0
    total_missing_lines: int = 0
    total_excluded_lines: int = 0
    total_branches: int = 0
    total_missing_branches: int = 0


@dataclass
class FileIssue:
    """Information about coverage issues in a specific file."""

    file_path: str
    line_coverage: float
    branch_coverage: float
    missing_lines_count: int
    missing_branches_count: int
    excluded_lines_count: int
    missing_lines: list[int]
    missing_branches: list[list[int]]
    excluded_lines: list[int]
    total_statements: int
    total_branches: int


def _calculate_branch_coverage_summary(
    totals: dict[str, Any],
) -> tuple[int, int, float]:
    """Calculate overall branch coverage statistics."""
    total_branches = totals.get("num_branches", 0)
    missing_branches = totals.get("missing_branches", 0)

    if total_branches > 0:
        overall_branch_coverage = (
            (total_branches - missing_branches) / total_branches
        ) * 100
    else:
        overall_branch_coverage = 100.0

    return total_branches, missing_branches, overall_branch_coverage


def _calculate_file_branch_coverage(
    file_summary: dict[str, Any], total_statements: int
) -> float:
    """Calculate branch coverage for a single file."""
    total_branches = file_summary.get("num_branches", 0)

    if total_branches > 0:
        covered_branches = file_summary.get("covered_branches", 0)
        return (covered_branches / total_branches) * 100

    return 100.0 if total_statements > 0 else 0.0


def _create_file_issue(
    file_path: str,
    file_summary: dict[str, Any],
    missing_lines: list[int],
    missing_branches: list[list[int]],
    excluded_lines: list[int],
    line_coverage: float,
    branch_coverage: float,
) -> FileIssue:
    """Create a FileIssue object for a file with coverage issues."""
    return FileIssue(
        file_path=file_path,
        line_coverage=line_coverage,
        branch_coverage=branch_coverage,
        missing_lines_count=len(missing_lines),
        missing_branches_count=len(missing_branches),
        excluded_lines_count=len(excluded_lines),
        missing_lines=missing_lines,
        missing_branches=missing_branches,
        excluded_lines=excluded_lines,
        total_statements=file_summary.get("num_statements", 0),
        total_branches=file_summary.get("num_branches", 0),
    )


def _analyze_single_file(
    file_path: str, file_data: dict[str, Any], summary: CoverageSummary
) -> FileIssue | None:
    """Analyze a single file and update summary, return FileIssue if needed."""
    summary.total_files += 1
    file_summary = file_data.get("summary", {})
    missing_lines = file_data.get("missing_lines", [])
    missing_branches = file_data.get("missing_branches", [])
    excluded_lines = file_data.get("excluded_lines", [])

    # Get file statistics
    total_statements = file_summary.get("num_statements", 0)
    line_coverage = file_summary.get("percent_covered", 0.0)
    branch_coverage = _calculate_file_branch_coverage(file_summary, total_statements)

    # Check if file has perfect coverage
    has_missing_lines = len(missing_lines) > 0
    has_missing_branches = len(missing_branches) > 0
    has_excluded_lines = len(excluded_lines) > 0

    if has_missing_lines:
        summary.files_with_missing_lines += 1
    if has_missing_branches:
        summary.files_with_missing_branches += 1
    if has_excluded_lines:
        summary.files_with_excluded_lines += 1

    if not has_missing_lines and not has_missing_branches:
        summary.files_with_perfect_coverage += 1

    # Return FileIssue if there are coverage issues (missing lines/branches)
    # OR if there are excluded lines to report (for informational purposes)
    if has_missing_lines or has_missing_branches or has_excluded_lines:
        return _create_file_issue(
            file_path,
            file_summary,
            missing_lines,
            missing_branches,
            excluded_lines,
            line_coverage,
            branch_coverage,
        )

    return None


def analyze_coverage_json(
    coverage_file: Path,
) -> tuple[CoverageSummary, list[FileIssue]]:
    """Analyze coverage data from a JSON file."""
    coverage_data = load_json_file(coverage_file, "coverage analysis")

    files_data = coverage_data.get("files", {})
    totals = coverage_data.get("totals", {})

    summary = CoverageSummary()

    # Set basic totals
    summary.total_statements = totals.get("num_statements", 0)
    summary.total_missing_lines = totals.get("missing_lines", 0)
    summary.total_excluded_lines = totals.get("excluded_lines", 0)
    summary.overall_line_coverage = totals.get("percent_covered", 0.0)

    # Calculate branch coverage from totals
    branch_data = _calculate_branch_coverage_summary(totals)
    total_branches, missing_branches, overall_branch_coverage = branch_data
    summary.total_branches = total_branches
    summary.total_missing_branches = missing_branches
    summary.overall_branch_coverage = overall_branch_coverage

    # Analyze each file
    file_issues = []
    for file_path, file_data in files_data.items():
        file_issue = _analyze_single_file(file_path, file_data, summary)
        if file_issue:
            file_issues.append(file_issue)

    # Sort file issues by line coverage (worst first)
    file_issues.sort(key=lambda x: (x.line_coverage, x.branch_coverage))

    return summary, file_issues


def print_coverage_summary(summary: CoverageSummary) -> None:
    """Print overall coverage summary."""
    print("=" * 80)
    print("COVERAGE ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    print("Overall Statistics:")
    print(f"  Total files analyzed: {summary.total_files}")
    print(f"  Files with perfect coverage: {summary.files_with_perfect_coverage}")
    print(f"  Files with missing lines: {summary.files_with_missing_lines}")
    print(f"  Files with missing branches: {summary.files_with_missing_branches}")
    print(f"  Files with excluded lines: {summary.files_with_excluded_lines}")
    print()

    print("Coverage Percentages:")
    print(f"  Overall line coverage: {summary.overall_line_coverage:.1f}%")
    print(f"  Overall branch coverage: {summary.overall_branch_coverage:.1f}%")
    print()

    print("Detailed Counts:")
    print(f"  Total statements: {summary.total_statements}")
    print(f"  Missing lines: {summary.total_missing_lines}")
    print(f"  Excluded lines: {summary.total_excluded_lines}")
    print(f"  Total branches: {summary.total_branches}")
    print(f"  Missing branches: {summary.total_missing_branches}")
    print()


def _print_source_code_for_missing_lines(issue: FileIssue) -> None:
    """Print source code for missing lines if file exists."""
    try:
        source_file = Path(issue.file_path)
        if source_file.exists():
            with open(source_file, encoding="utf-8") as f:
                lines = f.readlines()

            print("   Source code for missing lines:")
            for line_num in sorted(issue.missing_lines):
                if 1 <= line_num <= len(lines):
                    line_content = lines[line_num - 1].rstrip()
                    print(f"     {line_num:4d}: {line_content}")
        else:
            print("   (Source file not found for line details)")
    except (OSError, UnicodeDecodeError) as e:
        print(f"   (Error reading source file: {e})")


def _print_file_issue_details(issue: FileIssue, show_source: bool) -> None:
    """Print detailed coverage information for a single file."""
    # Show coverage percentages
    covered_statements = issue.total_statements - issue.missing_lines_count
    print(
        f"   Line Coverage: {issue.line_coverage:.1f}% "
        f"({covered_statements}/{issue.total_statements} statements)"
    )

    if issue.total_branches > 0:
        covered_branches = issue.total_branches - issue.missing_branches_count
        print(
            f"   Branch Coverage: {issue.branch_coverage:.1f}% "
            f"({covered_branches}/{issue.total_branches} branches)"
        )

    # Show missing lines
    if issue.missing_lines:
        lines_str = format_line_ranges(issue.missing_lines)
        print(f"   Missing Lines ({issue.missing_lines_count}): {lines_str}")

        if show_source:
            _print_source_code_for_missing_lines(issue)

    # Show missing branches
    if issue.missing_branches:
        branches_count = issue.missing_branches_count
        print(f"   Missing Branches ({branches_count}):")
        for branch in issue.missing_branches:
            print(f"     {branch}")

    # Show excluded lines
    if issue.excluded_lines:
        lines_str = format_line_ranges(issue.excluded_lines)
        print(f"   Excluded Lines ({issue.excluded_lines_count}): {lines_str}")


def print_file_issues(file_issues: list[FileIssue], show_source: bool = True) -> None:
    """Print detailed information about files with coverage issues."""
    if not file_issues:
        print("🎉 ALL FILES HAVE PERFECT COVERAGE!")
        return

    print("FILES WITH COVERAGE ISSUES OR EXCLUDED LINES:")
    print("=" * 80)
    print()

    for i, issue in enumerate(file_issues, 1):
        print(f"{i}. {issue.file_path}")
        print("-" * len(f"{i}. {issue.file_path}"))

        _print_file_issue_details(issue, show_source)
        print()


def print_perfect_coverage_files(
    file_issues: list[FileIssue], all_files: list[str]
) -> None:
    """Print files with perfect coverage."""
    files_with_issues = {issue.file_path for issue in file_issues}
    perfect_files = [f for f in all_files if f not in files_with_issues]

    if perfect_files:
        print("FILES WITH PERFECT COVERAGE:")
        print("=" * 80)
        for file_path in sorted(perfect_files):
            print(f"✅ {file_path}")
        print()


@click.command()
@click.argument("coverage_file", type=click.Path(exists=True, path_type=Path))
@click.option("--show-source", is_flag=True, help="Show source code for missing lines")
@click.option("--show-perfect", is_flag=True, help="Show files with perfect coverage")
def main(coverage_file: Path, show_source: bool, show_perfect: bool) -> None:
    """Analyze test coverage output in JSON format.

    This tool provides a detailed analysis of test coverage including
    overall coverage summary, per-file breakdown, detailed information
    about untested lines and branches, and files with imperfect coverage.

    COVERAGE_FILE  Path to the coverage JSON file to analyze
    """
    try:
        summary, file_issues = analyze_coverage_json(coverage_file)

        print_coverage_summary(summary)
        print_file_issues(file_issues, show_source=show_source)

        if show_perfect:
            # Get all file paths from the original data
            coverage_data = load_json_file(coverage_file, "coverage analysis")
            all_files = list(coverage_data.get("files", {}).keys())
            print_perfect_coverage_files(file_issues, all_files)

    except OSError as e:
        print(f"Error analyzing coverage file: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    # Click decorators inject arguments automatically from sys.argv
    main()  # pylint: disable=no-value-for-parameter
