#!/usr/bin/env python
"""List all available fctools and their descriptions."""

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from .ast_utils import format_docstring_summary, get_module_docstring
else:
    try:
        from .ast_utils import format_docstring_summary, get_module_docstring
    except ImportError:  # pragma: no cover
        from ast_utils import format_docstring_summary, get_module_docstring


def _extract_description_from_docstring(docstring: str) -> str:
    """Extract a brief description from a module's docstring.

    Takes the first line of the docstring, or falls back to a generic description.
    """
    if not docstring:
        return "Tool description not available"

    # Get the first non-empty line
    lines = docstring.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith('"""') and not line.startswith("'''"):
            return line

    return "Tool description not available"


def _extract_description_from_file(file_path: Path) -> str:
    """Extract description from a Python file's module docstring."""
    try:
        # Try to get the module docstring using ast_utils
        docstring = get_module_docstring(file_path)
        if docstring:
            return format_docstring_summary(docstring, max_length=80)

    except (OSError, SyntaxError, UnicodeDecodeError):
        # Fallback: try to import and get __doc__
        try:
            spec = importlib.util.spec_from_file_location("temp_module", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                doc = getattr(module, "__doc__", "")
                if doc:
                    return _extract_description_from_docstring(doc)
        except (ImportError, AttributeError, SyntaxError, FileNotFoundError):
            pass

    return "Tool description not available"


def _discover_tools() -> dict[str, str]:
    """Auto-discover all Python tools in the fctools directory."""
    tools = {}

    # Get the directory where this script lives (fctools/)
    fctools_dir = Path(__file__).parent

    # Find all .py files in the fctools directory
    for file_path in fctools_dir.glob("*.py"):
        filename = file_path.name

        # Skip special files
        if filename in ("__init__.py", "list_tools.py"):
            continue

        # Skip if not executable or doesn't look like a tool
        if filename.startswith("_"):
            continue

        description = _extract_description_from_file(file_path)
        tools[filename] = description

    return tools


app = typer.Typer()


@app.command()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed descriptions and usage examples",
    )
):
    """List all available fctools and their descriptions.

    This command helps discover what tools are available in the fctools
    package and provides quick access to their help information.

    Tools are automatically discovered by scanning the fctools directory
    for Python files and extracting their docstrings.
    """
    console = Console()
    console.print("Available fctools:\n")

    tools = _discover_tools()

    # Sort tools alphabetically for consistent output
    for tool in sorted(tools.keys()):
        description = tools[tool]

        if verbose:
            console.print(f"  [bold]{tool}[/bold]")
            console.print(f"    {description}")
            tool_name = tool.replace(".py", "")
            console.print(f"    Usage: python -m firecrown.fctools.{tool_name} --help")
            console.print()
        else:
            console.print(f"  [bold]{tool:<25}[/bold] - {description}")

    if not verbose:
        console.print("\nUse --verbose for detailed information about each tool.")
        console.print(
            "Use 'python -m firecrown.fctools.TOOL --help' for tool-specific help."
        )


if __name__ == "__main__":  # pragma: no cover
    app()
