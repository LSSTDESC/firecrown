# fctools - Development Tools

This directory contains command-line tools for code analysis, coverage reporting, and debugging.
All tools use [Click](https://click.palletsprojects.com/) for their command-line interface and provide comprehensive help.
These tools are intended for development and debugging purposes only.
They are *not* part of the Firecrown API and should not be used by any part of Firecrown (or by code that uses Firecrown).

Developers may use them for their own code analysis purposes, but they are not supported.

## Quick Start

List all available tools:

```bash
python -m firecrown.fctools.list_tools
```

Get detailed help for any tool:

```bash
python -m firecrown.fctools.TOOL --help
```

## Available Tools

### coverage_to_tsv.py

Convert pytest-cov JSON coverage data to TSV format. Supports merging timing data.

```bash
# Basic usage
python -m firecrown.fctools.coverage_to_tsv coverage.json

# With custom output and timing data
python -m firecrown.fctools.coverage_to_tsv coverage.json output.tsv --timing timing.json
```

### coverage_summary.py

Analyze test coverage output and provide detailed summaries.

```bash
# Basic summary
python -m firecrown.fctools.coverage_summary coverage.json

# Show source code for missing lines
python -m firecrown.fctools.coverage_summary coverage.json --show-source

# Include files with perfect coverage
python -m firecrown.fctools.coverage_summary coverage.json --show-perfect
```

### measurement_compatibility.py

Analyze measurement compatibility for Firecrown two-point functions. Shows which measurement pairs are compatible and provides insights for test optimization.

```bash
# Basic compatibility analysis
python -m firecrown.fctools.measurement_compatibility

# Detailed analysis with measurement lists
python -m firecrown.fctools.measurement_compatibility --verbose

# Analyze specific space only
python -m firecrown.fctools.measurement_compatibility --space real
python -m firecrown.fctools.measurement_compatibility --space harmonic

# Statistics only
python -m firecrown.fctools.measurement_compatibility --stats-only
```

### print_hierarchy.py

Display class hierarchy (Method Resolution Order) for Python types.

```bash
# Single class
python -m firecrown.fctools.print_hierarchy collections.OrderedDict

# Multiple classes
python -m firecrown.fctools.print_hierarchy pathlib.Path collections.Counter
```

### print_code.py

Display class definitions with attributes and decorators (excludes methods).

```bash
# With markdown formatting (default)
python -m firecrown.fctools.print_code dataclasses.dataclass

# Plain text output
python -m firecrown.fctools.print_code mymodule.MyClass --no-markdown

# Multiple classes
python -m firecrown.fctools.print_code class1 class2 class3
```

### tracer.py

Trace execution of Python scripts or modules, recording function calls to TSV.

```bash
# Trace a script
python -m firecrown.fctools.tracer myscript.py

# Trace with custom output file
python -m firecrown.fctools.tracer myscript.py --output trace_debug.tsv

# Trace a module
python -m firecrown.fctools.tracer mypackage.mymodule --module
```

## Features

- **Consistent CLI**: All tools use Click for professional command-line interfaces
- **Comprehensive help**: Every tool supports `--help` with detailed usage information  
- **Error handling**: Robust error handling with clear error messages
- **Code quality**: All tools pass flake8 linting (with acceptable complexity exceptions)
- **Python 3.13+**: Compatible with modern Python versions

## Development

All tools are standalone Python scripts that can be run directly or imported as modules. The `__init__.py` file provides comprehensive documentation of all available tools.

For tool discovery, use:

```bash
python -m firecrown.fctools.list_tools --verbose
```
