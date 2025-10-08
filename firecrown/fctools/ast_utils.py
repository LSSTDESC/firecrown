"""AST utilities for fctools.

This module provides utilities for parsing Python source code using the
Abstract Syntax Tree (AST) module, extracting docstrings, and analyzing
code structure.
"""

import ast
from pathlib import Path


def get_module_docstring(file_path: Path) -> str | None:
    """Extract the module-level docstring from a Python file.

    :param file_path: Path to the Python source file
    :return: The module docstring if present, None otherwise
    :raises OSError: If the file cannot be read
    :raises SyntaxError: If the file contains invalid Python syntax
    :raises UnicodeDecodeError: If the file encoding is invalid
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=str(file_path))
    return ast.get_docstring(tree)


def get_class_definition(source: str, class_name: str) -> ast.ClassDef | None:
    """Find a class definition node in Python source code.

    :param source: Python source code as a string
    :param class_name: Name of the class to find
    :return: The ClassDef AST node if found, None otherwise
    :raises SyntaxError: If the source contains invalid Python syntax
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def format_docstring_summary(docstring: str | None, max_length: int = 80) -> str:
    r"""Extract a brief summary from a docstring.

    Takes the first meaningful (non-empty, non-marker) line from the docstring.
    If the line is longer than max_length, it will be truncated with '...'.

    :param docstring: The docstring to summarize (may be None)
    :param max_length: Maximum length for the summary (default: 80)
    :return: A brief summary string, or a default message if no docstring

    Example usage::

        >>> format_docstring_summary("This is a tool.\\n\\nMore details.")
        'This is a tool.'
        >>> format_docstring_summary(None)
        'No description available'
    """
    if not docstring:
        return "No description available"

    # Get the first non-empty line, skipping quote markers
    lines = docstring.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Skip empty lines and quote markers
        if line and not line.startswith('"""') and not line.startswith("'''"):
            # Truncate if too long
            if len(line) > max_length:
                return line[: max_length - 3] + "..."
            return line

    return "No description available"


def extract_class_attributes(class_def: ast.ClassDef) -> list[str]:
    """Extract class-level attribute names from a ClassDef node.

    :param class_def: An AST ClassDef node
    :return: List of attribute names defined at class level

    .. note::
        This extracts simple assignments like `name: type` or `name = value`
        at the class body level, not instance attributes defined in __init__.
    """
    attributes = []

    for node in class_def.body:
        # Handle annotated assignments (e.g., name: type = value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            attributes.append(node.target.id)
        # Handle simple assignments (e.g., name = value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    attributes.append(target.id)

    return attributes


def format_class_docstring(class_def: ast.ClassDef) -> list[str]:
    """Extract and format the docstring from a ClassDef node.

    :param class_def: An AST ClassDef node
    :return: List of formatted docstring lines (empty list if no docstring)
    """
    docstring = ast.get_docstring(class_def)
    if not docstring:
        return []

    lines: list[str] = []
    for line in docstring.split("\n"):
        # Preserve indentation and content
        lines.append(f'    """{line}' if not lines else f"    {line}")

    # Close the docstring
    if lines:
        lines[-1] += '"""'

    return lines


def get_function_names(class_def: ast.ClassDef) -> list[str]:
    """Extract names of methods defined in a class.

    :param class_def: An AST ClassDef node
    :return: List of method names defined in the class

    .. note::
        This only returns methods defined directly in the class,
        not inherited methods.
    """
    methods = []
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef):
            methods.append(node.name)
    return methods
