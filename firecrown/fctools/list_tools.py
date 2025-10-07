#!/usr/bin/env python
"""List all available fctools and their descriptions."""

import ast
import importlib.util
from pathlib import Path

import click


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
        # Try to parse the AST to get the module docstring
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
        if docstring:
            return _extract_description_from_docstring(docstring)

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
        except (ImportError, AttributeError, SyntaxError):
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


@click.command()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed descriptions and usage examples",
)
def main(verbose: bool):
    """List all available fctools and their descriptions.

    This command helps discover what tools are available in the fctools
    package and provides quick access to their help information.

    Tools are automatically discovered by scanning the fctools directory
    for Python files and extracting their docstrings.
    """
    click.echo("Available fctools:\n")

    tools = _discover_tools()

    # Sort tools alphabetically for consistent output
    for tool in sorted(tools.keys()):
        description = tools[tool]

        if verbose:
            click.echo(f"  {tool}")
            click.echo(f"    {description}")
            tool_name = tool.replace(".py", "")
            click.echo(f"    Usage: python -m firecrown.fctools.{tool_name} --help")
            click.echo()
        else:
            click.echo(f"  {tool:<25} - {description}")

    if not verbose:
        click.echo("\nUse --verbose for detailed information about each tool.")
        click.echo(
            "Use 'python -m firecrown.fctools.TOOL --help' for tool-specific help."
        )


if __name__ == "__main__":
    # Click decorators inject arguments automatically from sys.argv
    main()  # pylint: disable=no-value-for-parameter
