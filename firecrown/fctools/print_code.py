"""Module for displaying class definitions with attributes and decorators.

This module provides utilities to inspect and display Python class
definitions in a formatted way suitable for syntax highlighting.
"""

import ast
import inspect
from typing import TYPE_CHECKING, Any, List

import typer
from rich import print as richprint
from rich.console import Console

if TYPE_CHECKING:
    from .ast_utils import format_class_docstring, get_class_definition
    from .common import import_class_from_path
else:
    try:
        from .ast_utils import format_class_docstring, get_class_definition
        from .common import import_class_from_path
    except ImportError:  # pragma: no cover
        from ast_utils import format_class_docstring, get_class_definition
        from common import import_class_from_path


def _render_attributes(class_def: ast.ClassDef) -> list[str]:
    """Render class attributes as formatted code lines."""
    lines: list[str] = []
    for item in class_def.body:
        if isinstance(item, (ast.AnnAssign, ast.Assign)):
            lines.append(f"    {ast.unparse(item)}")
    return lines


def _build_class_code(cls: type[Any]) -> str:
    source = inspect.getsource(cls)
    class_def = get_class_definition(source, cls.__name__)
    if class_def is None:
        raise ValueError(f"Could not find class definition for {cls.__name__}")

    code_lines: list[str] = []
    for decorator in class_def.decorator_list:
        code_lines.append(f"@{ast.unparse(decorator)}")

    bases_str = ", ".join(ast.unparse(base) for base in class_def.bases)
    code_lines.append(f"class {class_def.name}({bases_str}):")

    # Add formatted docstring
    docstring_lines = format_class_docstring(class_def)
    code_lines.extend(docstring_lines)

    code_lines.extend(_render_attributes(class_def))

    return "\n".join(code_lines)


def display_class_attributes(cls: type[Any]) -> None:
    """Display class definition with attributes and decorators.

    Formatted for syntax highlighting in Quarto/Jupyter.

    Args:
        cls: The class to display
    """
    try:
        code_str = _build_class_code(cls)
    except OSError:  # pragma: no cover
        # Defensive: inspect.getsource raises TypeError for built-ins, not OSError
        # OSError would require file read failure for an importable class (very rare)
        richprint(
            f"Source code not available for {cls.__name__} "
            f"(likely a built-in or C extension class)"
        )
        return

    richprint("```python")
    richprint(code_str)
    richprint("```")


def display_class_without_markdown(cls: type[Any]) -> None:
    """Display class definition without markdown code blocks.

    Same as display_class_attributes but outputs plain code without
    markdown wrapper for syntax highlighting.

    Args:
        cls: The class to display
    """
    try:
        code_str = _build_class_code(cls)
    except OSError:  # pragma: no cover
        # Defensive: inspect.getsource raises TypeError for built-ins, not OSError
        # OSError would require file read failure for an importable class (very rare)
        richprint(
            f"Source code not available for {cls.__name__} "
            f"(likely a built-in or C extension class)"
        )
        return

    richprint(code_str)


app = typer.Typer()


@app.command()
def main(
    class_names: List[str] = typer.Argument(
        ..., help="One or more fully qualified class names"
    ),
    no_markdown: bool = typer.Option(
        False, "--no-markdown", help="Output plain code without markdown code blocks"
    ),
):
    """Display class definitions with attributes and decorators.

    This tool inspects Python classes and displays their definitions
    in a formatted way, showing decorators, inheritance, docstrings,
    and class attributes (but not methods).
    """
    console = Console()
    # Select the appropriate display function based on the no_markdown flag
    if no_markdown:
        display_class_attributes_func = display_class_without_markdown
    else:
        display_class_attributes_func = display_class_attributes

    for class_name in class_names:
        try:
            cls = import_class_from_path(console, class_name)
            if len(class_names) > 1:
                print(f"\n{'=' * 60}")
                richprint(f"Class: {class_name}")
                richprint("=" * 60)
            display_class_attributes_func(cls)
            if len(class_names) > 1:
                richprint()
        except (ImportError, ValueError, AttributeError) as e:  # pragma: no cover
            # Defensive: import_class_from_path calls cli_error -> sys.exit(1)
            # So SystemExit is raised instead of ImportError/ValueError/AttributeError
            richprint(f"Could not import or display class {class_name}")
            richprint(f"Error message: {e}")
            if len(class_names) > 1:
                richprint()


if __name__ == "__main__":  # pragma: no cover
    app()
