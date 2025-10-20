"""Common utility functions for fctools.

This module provides shared functionality used across multiple fctools,
including JSON loading, module importing, and standardized error handling.
"""

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console


def load_json_file(
    console: Console, file_path: Path, error_context: str = "reading file"
) -> dict[str, Any]:
    """Load JSON file with standard error handling.

    :param console: The rich console object.
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
        cli_error(
            console, f"Failed to read file {file_path} while {error_context}: {e}"
        )
        raise AssertionError("Unreachable") from e  # pragma: no cover
    except json.JSONDecodeError as e:
        cli_error(console, f"Invalid JSON in {file_path} while {error_context}: {e}")
        raise AssertionError("Unreachable") from e  # pragma: no cover


def import_class_from_path(console: Console, full_path: str) -> type[Any]:
    """Import a class or type from a fully qualified module path.

    :param console: The rich console object.
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
    except ValueError as exc:
        cli_error(
            console,
            f"Invalid class path '{full_path}'. "
            "Expected format: 'module.submodule.ClassName'",
        )
        raise AssertionError("Unreachable") from exc  # pragma: no cover

    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        cli_error(console, f"Could not import module '{module_path}': {e}")
        raise AssertionError("Unreachable") from e  # pragma: no cover
    except AttributeError as e:
        cli_error(
            console, f"Class '{class_name}' not found in module '{module_path}': {e}"
        )
        raise AssertionError("Unreachable") from e  # pragma: no cover


def import_module_from_file(
    console: Console, file_path: Path, module_name: str = "temp_module"
) -> Any:
    """Import a Python module from a file path.

    This is useful for dynamically loading modules for inspection without
    requiring them to be on the Python path.

    :param console: The rich console object.
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
            cli_error(console, f"Could not create module spec from {file_path}")
            raise AssertionError("Unreachable")  # pragma: no cover

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (ImportError, OSError, SyntaxError) as e:
        cli_error(console, f"Failed to import module from {file_path}: {e}")
        raise AssertionError("Unreachable") from e  # pragma: no cover


def cli_error(console: Console, message: str, exit_code: int = 1) -> None:
    """Print error message to stderr and exit the program.

    :param console: The rich console object (unused, kept for API consistency).
    :param message: Error message to display
    :param exit_code: Exit code (default: 1)

    .. note::
        This function never returns; it always exits the program.
    """
    _ = console  # Unused but kept for API consistency
    # Add ERROR: prefix if not already present
    if not message.startswith("ERROR:"):
        message = f"ERROR: {message}"
    # Create a stderr console for error output
    err_console = Console(stderr=True)
    err_console.print(f"[bold red]{message}[/bold red]")
    sys.exit(exit_code)


def cli_warning(console: Console, message: str) -> None:
    """Print warning message to stderr without exiting.

    :param console: The rich console object (unused, kept for API consistency).
    :param message: Warning message to display
    """
    _ = console  # Unused but kept for API consistency
    # Add Warning: prefix if not already present
    if not message.startswith("Warning:"):
        message = f"Warning: {message}"
    # Create a stderr console for warning output
    err_console = Console(stderr=True)
    err_console.print(f"[yellow]{message}[/yellow]")


def validate_input_file(
    console: Console, file_path: Path, file_description: str = "Input file"
) -> None:
    """Validate that an input file exists and is readable.

    :param console: The rich console object.
    :param file_path: Path to the file to validate
    :param file_description: Description of the file for error messages
    :raises SystemExit: If the file doesn't exist or isn't readable (exits with code 1)
    """
    if not file_path.exists():
        cli_error(console, f"{file_description} not found: {file_path}")

    if not file_path.is_file():
        cli_error(console, f"{file_description} is not a regular file: {file_path}")

    # Check if readable by attempting to open
    try:
        with open(file_path, encoding="utf-8"):
            pass
    except OSError as e:
        cli_error(console, f"{file_description} is not readable: {e}")


def validate_output_path(
    console: Console, output_path: Path, overwrite: bool = False
) -> None:
    """Validate output path and check overwrite permissions.

    :param console: The rich console object.
    :param output_path: Path where output will be written
    :param overwrite: Whether overwriting existing files is allowed
    :raises SystemExit: If the file exists and overwrite is False (exits with code 1)
    """
    if output_path.exists() and not overwrite:
        cli_error(
            console,
            f"Output file '{output_path}' already exists. "
            "Use --overwrite to replace it.",
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
