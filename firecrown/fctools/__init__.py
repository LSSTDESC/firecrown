"""Tools for exploring code that is under development.

This package provides several command-line tools for analyzing code,
coverage data, and debugging. All tools use Click for their CLI interface
and provide comprehensive help.

Available Tools:
================

coverage_to_tsv.py
    Convert pytest-cov JSON coverage data to TSV format.
    Supports merging timing data from pytest --durations or pytest-json-report.

    Usage:
        python -m firecrown.fctools.coverage_to_tsv coverage.json [output.tsv]
            [--timing timing.json]

coverage_summary.py
    Analyze test coverage output in JSON format.
    Provides detailed coverage summary and per-file breakdown.

    Usage:
        python -m firecrown.fctools.coverage_summary coverage.json
            [--show-source] [--show-perfect]

print_hierarchy.py
    Print the class hierarchy (MRO) for Python types.
    Shows inheritance hierarchy and methods defined in each class.

    Usage:
        python -m firecrown.fctools.print_hierarchy fully.qualified.ClassName

print_code.py
    Display class definitions with attributes and decorators.
    Formatted for syntax highlighting, excludes methods.

    Usage:
        python -m firecrown.fctools.print_code fully.qualified.ClassName
            [--no-markdown]

tracer.py
    Trace execution of Python scripts or modules.
    Records function calls, returns, and exceptions to TSV file.

    Usage:
        python -m firecrown.fctools.tracer script.py [--output trace.tsv]
            [--module]

Quick Start:
============

1. Get help for any tool:
   python -m firecrown.fctools.TOOL --help

2. Analyze coverage data:
   python -m firecrown.fctools.coverage_summary coverage.json --show-source
   python -m firecrown.fctools.coverage_to_tsv coverage.json output.tsv

3. Inspect classes:
   python -m firecrown.fctools.print_hierarchy collections.OrderedDict
   python -m firecrown.fctools.print_code dataclasses.dataclass --no-markdown

4. Debug execution:
   python -m firecrown.fctools.tracer myscript.py --output debug_trace.tsv

All tools support the --help option for detailed usage information.
"""
