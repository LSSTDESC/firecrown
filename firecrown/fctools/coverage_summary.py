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

import typer
from rich.console import Console

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
    console: Console, coverage_file: Path
) -> tuple[CoverageSummary, list[FileIssue]]:
    """Analyze coverage data from a JSON file."""
    coverage_data = load_json_file(console, coverage_file, "coverage analysis")

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


def print_coverage_summary(console: Console, summary: CoverageSummary) -> None:
    """Print overall coverage summary."""
    console.print("[bold]=" * 80 + "[/bold]")
    console.print("[bold]COVERAGE ANALYSIS SUMMARY[/bold]")
    console.print("[bold]=" * 80 + "[/bold]")
    console.print()

    console.print("[bold]Overall Statistics:[/bold]")
    console.print(f"  Total files analyzed: {summary.total_files}")
    console.print(
        f"  Files with perfect coverage: {summary.files_with_perfect_coverage}"
    )
    console.print(f"  Files with missing lines: {summary.files_with_missing_lines}")
    console.print(
        f"  Files with missing branches: {summary.files_with_missing_branches}"
    )
    console.print(f"  Files with excluded lines: {summary.files_with_excluded_lines}")
    console.print()

    console.print("[bold]Coverage Percentages:[/bold]")
    console.print(f"  Overall line coverage: {summary.overall_line_coverage:.1f}%")
    console.print(f"  Overall branch coverage: {summary.overall_branch_coverage:.1f}%")
    console.print()

    console.print("[bold]Detailed Counts:[/bold]")
    console.print(f"  Total statements: {summary.total_statements}")
    console.print(f"  Missing lines: {summary.total_missing_lines}")
    console.print(f"  Excluded lines: {summary.total_excluded_lines}")
    console.print(f"  Total branches: {summary.total_branches}")
    console.print(f"  Missing branches: {summary.total_missing_branches}")
    console.print()


def _print_source_code_for_missing_lines(console: Console, issue: FileIssue) -> None:
    """Print source code for missing lines if file exists."""
    try:
        source_file = Path(issue.file_path)
        if source_file.exists():
            with open(source_file, encoding="utf-8") as f:
                lines = f.readlines()

            console.print("   [cyan]Source code for missing lines:[/cyan]")
            for line_num in sorted(issue.missing_lines):
                if 1 <= line_num <= len(lines):
                    line_content = lines[line_num - 1].rstrip()
                    console.print(f"     [red]{line_num:4d}: {line_content}[/red]")
        else:
            console.print(
                "   [yellow](Source file not found for line details)[/yellow]"
            )
    except (OSError, UnicodeDecodeError) as e:
        console.print(f"   [red](Error reading source file: {e})[/red]")


def _print_file_issue_details(
    console: Console, issue: FileIssue, show_source: bool
) -> None:
    """Print detailed coverage information for a single file."""
    # Show coverage percentages
    covered_statements = issue.total_statements - issue.missing_lines_count
    console.print(
        f"   Line Coverage: [bold]{issue.line_coverage:.1f}%[/bold] "
        f"({covered_statements}/{issue.total_statements} statements)"
    )

    if issue.total_branches > 0:
        covered_branches = issue.total_branches - issue.missing_branches_count
        console.print(
            f"   Branch Coverage: [bold]{issue.branch_coverage:.1f}%[/bold] "
            f"({covered_branches}/{issue.total_branches} branches)"
        )

    # Show missing lines
    if issue.missing_lines:
        lines_str = format_line_ranges(issue.missing_lines)
        console.print(
            f"   Missing Lines ({issue.missing_lines_count}): [red]{lines_str}[/red]"
        )

        if show_source:
            _print_source_code_for_missing_lines(console, issue)

    # Show missing branches
    if issue.missing_branches:
        branches_count = issue.missing_branches_count
        console.print(f"   Missing Branches ({branches_count}):")
        for branch in issue.missing_branches:
            console.print(f"     [yellow]{branch}[/yellow]")

    # Show excluded lines
    if issue.excluded_lines:
        lines_str = format_line_ranges(issue.excluded_lines)
        console.print(
            f"   Excluded Lines ({issue.excluded_lines_count}): [dim]{lines_str}[/dim]"
        )


def print_file_issues(
    console: Console, file_issues: list[FileIssue], show_source: bool = True
) -> None:
    """Print detailed information about files with coverage issues."""
    if not file_issues:
        console.print("🎉 [bold green]ALL FILES HAVE PERFECT COVERAGE![/bold green]")
        return

    console.print("[bold]FILES WITH COVERAGE ISSUES OR EXCLUDED LINES:[/bold]")
    console.print("[bold]=" * 80 + "[/bold]")
    console.print()

    for i, issue in enumerate(file_issues, 1):
        console.print(f"[bold]{i}. {issue.file_path}[/bold]")
        console.print("-" * len(f"{i}. {issue.file_path}"))

        _print_file_issue_details(console, issue, show_source)
        console.print()


def print_perfect_coverage_files(
    console: Console, file_issues: list[FileIssue], all_files: list[str]
) -> None:
    """Print files with perfect coverage."""
    files_with_issues = {issue.file_path for issue in file_issues}
    perfect_files = [f for f in all_files if f not in files_with_issues]

    if perfect_files:
        console.print("[bold]FILES WITH PERFECT COVERAGE:[/bold]")
        console.print("[bold]=" * 80 + "[/bold]")
        for file_path in sorted(perfect_files):
            console.print(f"✅ [green]{file_path}[/green]")
        console.print()


app = typer.Typer()


@app.command()
def main(
    coverage_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the coverage JSON file to analyze",
    ),
    show_source: bool = typer.Option(
        False, "--show-source", help="Show source code for missing lines"
    ),
    show_perfect: bool = typer.Option(
        False, "--show-perfect", help="Show files with perfect coverage"
    ),
) -> None:
    """Analyze test coverage output in JSON format.

    This tool provides a detailed analysis of test coverage including
    overall coverage summary, per-file breakdown, detailed information
    about untested lines and branches, and files with imperfect coverage.
    """
    console = Console()
    try:
        summary, file_issues = analyze_coverage_json(console, coverage_file)

        print_coverage_summary(console, summary)
        print_file_issues(console, file_issues, show_source=show_source)

        if show_perfect:
            # Get all file paths from the original data
            coverage_data = load_json_file(console, coverage_file, "coverage analysis")
            all_files = list(coverage_data.get("files", {}).keys())
            print_perfect_coverage_files(console, file_issues, all_files)

    except OSError as e:
        console.print(f"[bold red]Error analyzing coverage file: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()
