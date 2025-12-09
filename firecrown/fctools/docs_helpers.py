"""Helper functions shared by documentation validation tools.

Provides common functionality for file/directory validation, JSON loading,
and rich console output for code_block_checker, symbol_reference_checker,
and generate_symbol_map tools.
"""

import json
import sys
from pathlib import Path

from rich.console import Console


def load_json_file(file_path: Path, error_prefix: str = "JSON file") -> dict:
    """Load and parse a JSON file with error handling.

    :param file_path: Path to the JSON file
    :param error_prefix: Prefix for error messages
    :return: Parsed JSON as dictionary
    :raises typer.Exit: If file cannot be loaded or parsed
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        console = Console(stderr=True)
        console.print(f"[red]Error: Failed to parse {error_prefix}: {e}[/red]")
        sys.exit(1)
    except OSError as e:
        console = Console(stderr=True)
        console.print(f"[red]Error: Could not read {error_prefix}: {e}[/red]")
        sys.exit(1)


def validate_directory_exists(directory: Path, error_prefix: str = "Directory") -> None:
    """Validate that a directory exists.

    :param directory: Path to validate
    :param error_prefix: Prefix for error message
    :raises typer.Exit: If directory doesn't exist
    """
    if not directory.is_dir():
        console = Console(stderr=True)
        console.print(f"[red]Error: {error_prefix} not found at '{directory}'[/red]")
        sys.exit(1)


def validate_file_exists(file_path: Path, error_prefix: str = "File") -> None:
    """Validate that a file exists.

    :param file_path: Path to validate
    :param error_prefix: Prefix for error message
    :raises typer.Exit: If file doesn't exist
    """
    if not file_path.is_file():
        console = Console(stderr=True)
        console.print(f"[red]Error: {error_prefix} not found at '{file_path}'[/red]")
        sys.exit(1)


def print_success(message: str) -> None:
    """Print a success message with green checkmark.

    :param message: Message to print
    """
    console = Console()
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message in red.

    :param message: Error message to print
    """
    console = Console(stderr=True)
    console.print(f"[red]✗[/red] {message}")
