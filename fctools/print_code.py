"""
Module for displaying class definitions with attributes and decorators.

This module provides utilities to inspect and display Python class
definitions in a formatted way suitable for syntax highlighting.
"""

import ast
import click
import importlib
import inspect
from typing import Type, Any


def display_class_attributes(cls: Type[Any]) -> None:
    """Display class definition with attributes and decorators.

    Formatted for syntax highlighting in Quarto/Jupyter.

    Args:
        cls: The class to display
    """
    try:
        # Get the source code of the class
        source = inspect.getsource(cls)
    except OSError:
        print(
            f"Source code not available for {cls.__name__} "
            f"(likely a built-in or C extension class)"
        )
        return

    # Parse into an AST
    parsed = ast.parse(source)

    # Find the class definition node
    class_def = None
    for node in ast.iter_child_nodes(parsed):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            class_def = node
            break

    if not class_def:
        print(f"Could not find class definition for {cls.__name__}")
        return

    # Build the output code as a string
    code_lines = []

    # Add decorators
    for decorator in class_def.decorator_list:
        code_lines.append(f"@{ast.unparse(decorator)}")

    # Add class definition line
    bases_str = ", ".join(ast.unparse(base) for base in class_def.bases)
    code_lines.append(f"class {class_def.name}({bases_str}):")

    # Add docstring if present
    if (
        class_def.body
        and isinstance(class_def.body[0], ast.Expr)
        and isinstance(class_def.body[0].value, ast.Constant)
        and isinstance(class_def.body[0].value.value, str)
    ):
        docstring = class_def.body[0].value.value
        if "\n" in docstring:
            code_lines.append('    """')
            for line in docstring.strip().split("\n"):
                code_lines.append(f"    {line}")
            code_lines.append('    """')
        else:
            code_lines.append(f'    """{docstring}"""')

    # Add attributes but skip methods
    for item in class_def.body:
        if isinstance(item, (ast.AnnAssign, ast.Assign)):
            code_lines.append(f"    {ast.unparse(item)}")

    # Join all lines into a single string
    code_str = "\n".join(code_lines)

    # Output with markdown code block syntax for syntax highlighting
    print("```python")
    print(code_str)
    print("```")


def import_class(full_path: str) -> Type[Any]:
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
    # Temporarily modify display function if no-markdown requested
    if no_markdown:
        global _original_display_class_attributes
        _original_display_class_attributes = display_class_attributes

        def display_without_markdown(cls: Type[Any]) -> None:
            # Same logic but without markdown wrapper
            try:
                source = inspect.getsource(cls)
            except OSError:
                print(
                    f"Source code not available for {cls.__name__} "
                    f"(likely a built-in or C extension class)"
                )
                return
            parsed = ast.parse(source)
            class_def = None
            for node in ast.iter_child_nodes(parsed):
                if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                    class_def = node
                    break

            if not class_def:
                print(f"Could not find class definition for {cls.__name__}")
                return

            code_lines = []
            for decorator in class_def.decorator_list:
                code_lines.append(f"@{ast.unparse(decorator)}")

            bases_str = ", ".join(ast.unparse(base) for base in class_def.bases)
            code_lines.append(f"class {class_def.name}({bases_str}):")

            if (
                class_def.body
                and isinstance(class_def.body[0], ast.Expr)
                and isinstance(class_def.body[0].value, ast.Constant)
                and isinstance(class_def.body[0].value.value, str)
            ):
                docstring = class_def.body[0].value.value
                if "\n" in docstring:
                    code_lines.append('    """')
                    for line in docstring.strip().split("\n"):
                        code_lines.append(f"    {line}")
                    code_lines.append('    """')
                else:
                    code_lines.append(f'    """{docstring}"""')

            for item in class_def.body:
                if isinstance(item, (ast.AnnAssign, ast.Assign)):
                    code_lines.append(f"    {ast.unparse(item)}")

            code_str = "\n".join(code_lines)
            print(code_str)

        # Temporarily replace the function
        display_class_attributes_func = display_without_markdown
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
    main()
