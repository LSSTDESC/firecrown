#!/usr/bin/env python
"""
Comprehensive tool to analyze test coverage output in JSON format.

This tool provides a detailed analysis of test coverage including:
- Overall coverage summary
- Per-file coverage breakdown
- Detailed information about untested lines and branches
- Files with less than perfect coverage
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CoverageSummary:
    """Summary statistics for coverage analysis."""

    total_files: int = 0
    files_with_perfect_coverage: int = 0
    files_with_missing_lines: int = 0
    files_with_missing_branches: int = 0
    overall_line_coverage: float = 0.0
    overall_branch_coverage: float = 0.0
    total_statements: int = 0
    total_missing_lines: int = 0
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
    missing_lines: List[int]
    missing_branches: List[List[int]]
    total_statements: int
    total_branches: int


def group_consecutive_lines(lines: List[int]) -> List[str]:
    """Group consecutive line numbers for better readability."""
    if not lines:
        return []

    sorted_lines = sorted(lines)
    groups = []
    current_group = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        if line == current_group[-1] + 1:
            current_group.append(line)
        else:
            groups.append(current_group)
            current_group = [line]
    groups.append(current_group)

    result = []
    for group in groups:
        if len(group) == 1:
            result.append(f"{group[0]}")
        else:
            result.append(f"{group[0]}-{group[-1]}")

    return result


def analyze_coverage_json(
    coverage_file: Path,
) -> tuple[CoverageSummary, List[FileIssue]]:
    """Analyze coverage data from a JSON file."""
    with open(coverage_file, "r") as f:
        coverage_data = json.load(f)

    files_data = coverage_data.get("files", {})
    totals = coverage_data.get("totals", {})

    summary = CoverageSummary()
    file_issues = []

    # Overall totals
    summary.total_statements = totals.get("num_statements", 0)
    summary.total_missing_lines = totals.get("missing_lines", 0)
    summary.overall_line_coverage = totals.get("percent_covered", 0.0)

    # Calculate branch coverage from totals
    total_branches = totals.get("num_branches", 0)
    missing_branches = totals.get("missing_branches", 0)
    summary.total_branches = total_branches
    summary.total_missing_branches = missing_branches

    if total_branches > 0:
        summary.overall_branch_coverage = (
            (total_branches - missing_branches) / total_branches
        ) * 100
    else:
        summary.overall_branch_coverage = 100.0

    # Analyze each file
    for file_path, file_data in files_data.items():
        summary.total_files += 1

        file_summary = file_data.get("summary", {})
        missing_lines = file_data.get("missing_lines", [])
        missing_branches = file_data.get("missing_branches", [])

        # Get file statistics
        total_statements = file_summary.get("num_statements", 0)
        total_branches = file_summary.get("num_branches", 0)
        line_coverage = file_summary.get("percent_covered", 0.0)

        # Calculate branch coverage for this file
        if total_branches > 0:
            covered_branches = file_summary.get("covered_branches", 0)
            branch_coverage = (covered_branches / total_branches) * 100
        else:
            branch_coverage = 100.0 if total_statements > 0 else 0.0

        # Check if file has perfect coverage
        has_missing_lines = len(missing_lines) > 0
        has_missing_branches = len(missing_branches) > 0

        if has_missing_lines:
            summary.files_with_missing_lines += 1
        if has_missing_branches:
            summary.files_with_missing_branches += 1

        if not has_missing_lines and not has_missing_branches:
            summary.files_with_perfect_coverage += 1

        # Record issues for files with imperfect coverage
        if has_missing_lines or has_missing_branches:
            file_issue = FileIssue(
                file_path=file_path,
                line_coverage=line_coverage,
                branch_coverage=branch_coverage,
                missing_lines_count=len(missing_lines),
                missing_branches_count=len(missing_branches),
                missing_lines=missing_lines,
                missing_branches=missing_branches,
                total_statements=total_statements,
                total_branches=total_branches,
            )
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
    print()

    print("Coverage Percentages:")
    print(f"  Overall line coverage: {summary.overall_line_coverage:.1f}%")
    print(f"  Overall branch coverage: {summary.overall_branch_coverage:.1f}%")
    print()

    print("Detailed Counts:")
    print(f"  Total statements: {summary.total_statements}")
    print(f"  Missing lines: {summary.total_missing_lines}")
    print(f"  Total branches: {summary.total_branches}")
    print(f"  Missing branches: {summary.total_missing_branches}")
    print()


def print_file_issues(file_issues: List[FileIssue], show_source: bool = True) -> None:
    """Print detailed information about files with coverage issues."""
    if not file_issues:
        print("🎉 ALL FILES HAVE PERFECT COVERAGE!")
        return

    print("FILES WITH COVERAGE ISSUES:")
    print("=" * 80)
    print()

    for i, issue in enumerate(file_issues, 1):
        print(f"{i}. {issue.file_path}")
        print("-" * len(f"{i}. {issue.file_path}"))

        print(
            f"   Line Coverage: {issue.line_coverage:.1f}% "
            f"({issue.total_statements - issue.missing_lines_count}/{issue.total_statements} statements)"
        )

        if issue.total_branches > 0:
            print(
                f"   Branch Coverage: {issue.branch_coverage:.1f}% "
                f"({issue.total_branches - issue.missing_branches_count}/{issue.total_branches} branches)"
            )

        # Show missing lines
        if issue.missing_lines:
            line_groups = group_consecutive_lines(issue.missing_lines)
            print(
                f"   Missing Lines ({issue.missing_lines_count}): {', '.join(line_groups)}"
            )

            # Show source code for missing lines if requested and file exists
            if show_source:
                try:
                    source_file = Path(issue.file_path)
                    if source_file.exists():
                        with open(source_file, "r") as f:
                            lines = f.readlines()

                        print("   Source code for missing lines:")
                        for line_num in sorted(issue.missing_lines):
                            if 1 <= line_num <= len(lines):
                                line_content = lines[line_num - 1].rstrip()
                                print(f"     {line_num:4d}: {line_content}")
                    else:
                        print("   (Source file not found for line details)")
                except Exception as e:
                    print(f"   (Error reading source file: {e})")

        # Show missing branches
        if issue.missing_branches:
            print(f"   Missing Branches ({issue.missing_branches_count}):")
            for branch in issue.missing_branches:
                print(f"     {branch}")

        print()


def print_perfect_coverage_files(
    file_issues: List[FileIssue], all_files: List[str]
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


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python coverage_summary.py <coverage.json> [--show-source] [--show-perfect]"
        )
        print()
        print("Options:")
        print("  --show-source   Show source code for missing lines")
        print("  --show-perfect  Show files with perfect coverage")
        print()
        print("Example: python coverage_summary.py coverage.json --show-source")
        sys.exit(1)

    coverage_file = Path(sys.argv[1])
    show_source = "--show-source" in sys.argv
    show_perfect = "--show-perfect" in sys.argv

    if not coverage_file.exists():
        print(f"Error: Coverage file '{coverage_file}' not found.")
        sys.exit(1)

    try:
        summary, file_issues = analyze_coverage_json(coverage_file)

        print_coverage_summary(summary)
        print_file_issues(file_issues, show_source=show_source)

        if show_perfect:
            # Get all file paths from the original data
            with open(coverage_file, "r") as f:
                coverage_data = json.load(f)
            all_files = list(coverage_data.get("files", {}).keys())
            print_perfect_coverage_files(file_issues, all_files)

    except Exception as e:
        print(f"Error analyzing coverage file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
