import ast
import inspect
from typing import Type, Any


def display_class_attributes(cls: Type[Any]) -> None:
    """
    Display class definition with attributes and decorators,
    formatted for syntax highlighting in Quarto/Jupyter.

    Args:
        cls: The class to display
    """
    # Get the source code of the class
    source = inspect.getsource(cls)

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
        if isinstance(item, ast.AnnAssign) or isinstance(item, ast.Assign):
            code_lines.append(f"    {ast.unparse(item)}")

    # Join all lines into a single string
    code_str = "\n".join(code_lines)

    # Output with markdown code block syntax for syntax highlighting
    print("```python")
    print(code_str)
    print("```")
