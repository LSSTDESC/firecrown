"""Tools for exploring code that is under development.

This package provides several command-line tools for analyzing code,
coverage data, and debugging. All tools use Click for their CLI interface
and provide comprehensive help.

Available Tools:
================

coverage_to_tsv.py
    Convert pytest-cov JSON coverage data to TSV format.
    Supports merging timing data from pytest --durations or pytest-json-report.

    Usage: python fctools/coverage_to_tsv.py coverage.json [output.tsv] [--timing timing.json]

coverage_summary.py
    Analyze test coverage output in JSON format.
    Provides detailed coverage summary and per-file breakdown.

    Usage: python fctools/coverage_summary.py coverage.json [--show-source] [--show-perfect]

print_hierarchy.py
    Print the class hierarchy (MRO) for Python types.
    Shows inheritance hierarchy and methods defined in each class.

    Usage: python fctools/print_hierarchy.py fully.qualified.ClassName [...]

print_code.py
    Display class definitions with attributes and decorators.
    Formatted for syntax highlighting, excludes methods.

    Usage: python fctools/print_code.py fully.qualified.ClassName [...] [--no-markdown]

tracer.py
    Trace execution of Python scripts or modules.
    Records function calls, returns, and exceptions to TSV file.

    Usage: python fctools/tracer.py script.py [--output trace.tsv] [--module]

Quick Start:
============

1. Get help for any tool:
   python fctools/TOOL.py --help

2. Analyze coverage data:
   python fctools/coverage_summary.py coverage.json --show-source
   python fctools/coverage_to_tsv.py coverage.json output.tsv

3. Inspect classes:
   python fctools/print_hierarchy.py collections.OrderedDict
   python fctools/print_code.py dataclasses.dataclass --no-markdown

4. Debug execution:
   python fctools/tracer.py myscript.py --output debug_trace.tsv

All tools support the --help option for detailed usage information.
"""
