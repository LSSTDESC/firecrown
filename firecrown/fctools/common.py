"""Common utility functions for fctools.

This module provides shared functionality used across multiple fctools,
including JSON loading, module importing, and standardized error handling.
"""

import importlib
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import click


def load_json_file(
    file_path: Path, error_context: str = "reading file"
) -> dict[str, Any]:
    """Load JSON file with standard error handling.

    :param file_path: Path to the JSON file to load
    :param error_context: Context description for error messages
    :return: The loaded JSON data as a dictionary
    :raises SystemExit: If the file cannot be read or parsed (exits with code 1)

    .. note::
        This function will exit the program on error rather than raising exceptions,
        as it's designed for CLI tools that should fail gracefully.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        cli_error(f"Failed to read file {file_path} while {error_context}: {e}")
    except json.JSONDecodeError as e:
        cli_error(f"Invalid JSON in {file_path} while {error_context}: {e}")


def import_class_from_path(full_path: str) -> type[Any]:
    """Import a class or type from a fully qualified module path.

    :param full_path: Fully qualified path to the class (e.g., 'mymodule.MyClass')
    :return: The imported class/type object
    :raises SystemExit: If the module or class cannot be imported (exits with code 1)

    Example usage::

        >>> cls = import_class_from_path('pathlib.Path')
        >>> isinstance(cls, type)
        True
    """
    try:
        module_path, class_name = full_path.rsplit(".", 1)
    except ValueError:
        cli_error(
            f"Invalid class path '{full_path}'. "
            "Expected format: 'module.submodule.ClassName'"
        )

    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        cli_error(f"Could not import module '{module_path}': {e}")
    except AttributeError as e:
        cli_error(f"Class '{class_name}' not found in module '{module_path}': {e}")


def import_module_from_file(file_path: Path, module_name: str = "temp_module") -> Any:
    """Import a Python module from a file path.

    This is useful for dynamically loading modules for inspection without
    requiring them to be on the Python path.

    :param file_path: Path to the Python file to import
    :param module_name: Name to give the imported module (default: 'temp_module')
    :return: The imported module object
    :raises SystemExit: If the module cannot be imported (exits with code 1)

    .. note::
        This function executes the module code, so use with caution on untrusted files.
    """
    try:
        # pylint: disable=import-outside-toplevel,redefined-outer-name
        import importlib.util

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            cli_error(f"Could not create module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (ImportError, OSError, SyntaxError) as e:
        cli_error(f"Failed to import module from {file_path}: {e}")


def cli_error(message: str, exit_code: int = 1) -> NoReturn:
    """Print error message to stderr and exit the program.

    :param message: Error message to display
    :param exit_code: Exit code to use (default: 1)

    .. note::
        This function never returns; it always exits the program.
    """
    # Add ERROR: prefix if not already present
    if not message.startswith("ERROR:"):
        message = f"ERROR: {message}"
    click.echo(message, err=True)
    sys.exit(exit_code)


def cli_warning(message: str) -> None:
    """Print warning message to stderr without exiting.

    :param message: Warning message to display
    """
    # Add Warning: prefix if not already present
    if not message.startswith("Warning:"):
        message = f"Warning: {message}"
    click.echo(message, err=True)


def validate_input_file(file_path: Path, file_description: str = "Input file") -> None:
    """Validate that an input file exists and is readable.

    :param file_path: Path to the file to validate
    :param file_description: Description of the file for error messages
    :raises SystemExit: If the file doesn't exist or isn't readable (exits with code 1)
    """
    if not file_path.exists():
        cli_error(f"{file_description} not found: {file_path}")

    if not file_path.is_file():
        cli_error(f"{file_description} is not a regular file: {file_path}")

    # Check if readable by attempting to open
    try:
        with open(file_path, encoding="utf-8"):
            pass
    except OSError as e:
        cli_error(f"{file_description} is not readable: {e}")


def validate_output_path(output_path: Path, overwrite: bool = False) -> None:
    """Validate output path and check overwrite permissions.

    :param output_path: Path where output will be written
    :param overwrite: Whether overwriting existing files is allowed
    :raises SystemExit: If the file exists and overwrite is False (exits with code 1)
    """
    if output_path.exists() and not overwrite:
        cli_error(
            f"Output file '{output_path}' already exists. "
            "Use --overwrite to replace it."
        )


def format_line_ranges(lines: list[int]) -> list[str]:
    """Group consecutive line numbers into readable ranges.

    :param lines: List of line numbers
    :return: List of formatted strings representing line ranges

    Example usage::

        >>> format_line_ranges([1, 2, 3, 5, 6, 8])
        ['1-3', '5-6', '8']
        >>> format_line_ranges([10])
        ['10']
        >>> format_line_ranges([])
        []
    """
    if not lines:
        return []

    sorted_lines = sorted(lines)
    groups: list[list[int]] = []
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
