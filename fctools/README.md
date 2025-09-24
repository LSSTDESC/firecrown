# fctools - Development Tools

This directory contains command-line tools for code analysis, coverage reporting, and debugging. All tools use [Click](https://click.palletsprojects.com/) for their command-line interface and provide comprehensive help.

## Quick Start

List all available tools:

```bash
python fctools/list_tools.py
```

Get detailed help for any tool:

```bash
python fctools/TOOL.py --help
```

## Available Tools

### coverage_to_tsv.py

Convert pytest-cov JSON coverage data to TSV format. Supports merging timing data.

```bash
# Basic usage
python fctools/coverage_to_tsv.py coverage.json

# With custom output and timing data
python fctools/coverage_to_tsv.py coverage.json output.tsv --timing timing.json
```

### coverage_summary.py

Analyze test coverage output and provide detailed summaries.

```bash
# Basic summary
python fctools/coverage_summary.py coverage.json

# Show source code for missing lines
python fctools/coverage_summary.py coverage.json --show-source

# Include files with perfect coverage
python fctools/coverage_summary.py coverage.json --show-perfect
```

### print_hierarchy.py

Display class hierarchy (Method Resolution Order) for Python types.

```bash
# Single class
python fctools/print_hierarchy.py collections.OrderedDict

# Multiple classes
python fctools/print_hierarchy.py pathlib.Path collections.Counter
```

### print_code.py

Display class definitions with attributes and decorators (excludes methods).

```bash
# With markdown formatting (default)
python fctools/print_code.py dataclasses.dataclass

# Plain text output
python fctools/print_code.py mymodule.MyClass --no-markdown

# Multiple classes
python fctools/print_code.py class1 class2 class3
```

### tracer.py

Trace execution of Python scripts or modules, recording function calls to TSV.

```bash
# Trace a script
python fctools/tracer.py myscript.py

# Trace with custom output file
python fctools/tracer.py myscript.py --output trace_debug.tsv

# Trace a module
python fctools/tracer.py mypackage.mymodule --module
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
python fctools/list_tools.py --verbose
```
