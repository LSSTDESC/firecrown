"""Module for displaying class definitions with attributes and decorators.

This module provides utilities to inspect and display Python class
definitions in a formatted way suitable for syntax highlighting.
"""

import ast
import importlib
import inspect
from typing import Any

import click


def _get_class_def_node(source: str, cls_name: str) -> ast.ClassDef | None:
    parsed = ast.parse(source)
    for node in ast.iter_child_nodes(parsed):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return node
    return None


def _render_docstring(class_def: ast.ClassDef) -> list[str]:
    if (
        class_def.body
        and isinstance(class_def.body[0], ast.Expr)
        and isinstance(class_def.body[0].value, ast.Constant)
        and isinstance(class_def.body[0].value.value, str)
    ):
        docstring = class_def.body[0].value.value
        lines: list[str] = []
        if "\n" in docstring:
            lines.append('    """')
            for line in docstring.strip().split("\n"):
                lines.append(f"    {line}")
            lines.append('    """')
        else:
            lines.append(f'    """{docstring}"""')
        return lines
    return []


def _render_attributes(class_def: ast.ClassDef) -> list[str]:
    lines: list[str] = []
    for item in class_def.body:
        if isinstance(item, (ast.AnnAssign, ast.Assign)):
            lines.append(f"    {ast.unparse(item)}")
    return lines


def _build_class_code(cls: type[Any]) -> str:
    source = inspect.getsource(cls)
    class_def = _get_class_def_node(source, cls.__name__)
    if class_def is None:
        raise ValueError(f"Could not find class definition for {cls.__name__}")

    code_lines: list[str] = []
    for decorator in class_def.decorator_list:
        code_lines.append(f"@{ast.unparse(decorator)}")

    bases_str = ", ".join(ast.unparse(base) for base in class_def.bases)
    code_lines.append(f"class {class_def.name}({bases_str}):")
    code_lines.extend(_render_docstring(class_def))
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
    except OSError:
        print(
            f"Source code not available for {cls.__name__} "
            f"(likely a built-in or C extension class)"
        )
        return

    print("```python")
    print(code_str)
    print("```")


def display_class_without_markdown(cls: type[Any]) -> None:
    """Display class definition without markdown code blocks.

    Same as display_class_attributes but outputs plain code without
    markdown wrapper for syntax highlighting.

    Args:
        cls: The class to display
    """
    try:
        code_str = _build_class_code(cls)
    except OSError:
        print(
            f"Source code not available for {cls.__name__} "
            f"(likely a built-in or C extension class)"
        )
        return

    print(code_str)


def import_class(full_path: str) -> type[Any]:
    """Import a class from a full path, returning the class."""
    module_path, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not inspect.isclass(cls):
        raise ValueError(f"{full_path} is not a class")
    return cls


@click.command()
@click.argument("class_names", nargs=-1, required=True)
@click.option(
    "--no-markdown", is_flag=True, help="Output plain code without markdown code blocks"
)
def main(class_names, no_markdown: bool):
    """Display class definitions with attributes and decorators.

    This tool inspects Python classes and displays their definitions
    in a formatted way, showing decorators, inheritance, docstrings,
    and class attributes (but not methods).

    CLASS_NAMES  One or more fully qualified class names
    """
    # Select the appropriate display function based on the no_markdown flag
    if no_markdown:
        display_class_attributes_func = display_class_without_markdown
    else:
        display_class_attributes_func = display_class_attributes

    for class_name in class_names:
        try:
            cls = import_class(class_name)
            if len(class_names) > 1:
                print(f"\n{'=' * 60}")
                print(f"Class: {class_name}")
                print("=" * 60)
            display_class_attributes_func(cls)
            if len(class_names) > 1:
                print()
        except (ImportError, ValueError, AttributeError) as e:
            print(f"Could not import or display class {class_name}")
            print(f"Error message: {e}")
            if len(class_names) > 1:
                print()


if __name__ == "__main__":
    # Click decorators inject arguments automatically from sys.argv
    main()  # pylint: disable=no-value-for-parameter
