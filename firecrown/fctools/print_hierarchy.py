#!/usr/bin/env python
"""This script provides a way to print the class hierarchy of a given type."""

import inspect
from typing import TYPE_CHECKING, List

import typer
from rich import print as richprint
from rich.console import Console

if TYPE_CHECKING:
    from .common import import_class_from_path
else:
    try:
        from .common import import_class_from_path
    except ImportError:  # pragma: no cover
        from common import import_class_from_path


def full_type_name(t: type) -> str:
    """Return the full type name, with the module path to the type."""
    return f"{t.__module__}.{t.__name__}"


def get_defined_methods(cls) -> list[str]:
    """Return a list of method names defined in the class.

    This includes methods introduced in this class, as well as
    inherited methods that are overridden in this class. It does
    not include methods inherited from base classes.
    """
    cls_methods = dict(inspect.getmembers(cls, predicate=inspect.isfunction))
    inherited_methods = {}
    for base in cls.__bases__:
        inherited_methods.update(
            dict(inspect.getmembers(base, predicate=inspect.isfunction))
        )

    defined_methods = {
        name: obj
        for name, obj in cls_methods.items()
        if name not in inherited_methods or obj != inherited_methods[name]
    }
    return list(defined_methods.keys())


def print_one_type(idx: int, t: type) -> None:
    """Print the direct type information for the type t.

    This function does not traverse the type hierarchy; see
    print_type_hierarchy for that purpose.
    """
    richprint(f"{idx}  {full_type_name(t)}")
    methods = get_defined_methods(t)
    for method in methods:
        richprint(f"       {method}")


def print_type_hierarchy(top_type: type) -> None:
    """Print the class hierarchy for the given type."""
    richprint(f"Hierarchy for {full_type_name(top_type)}:")
    for i, t in enumerate(inspect.getmro(top_type)):
        print_one_type(i, t)


app = typer.Typer()


@app.command()
def main(
    typenames: List[str] = typer.Argument(
        ..., help="One or more fully qualified type names"
    )
):
    """Print the class hierarchy for the given type(s).

    This tool displays the Method Resolution Order (MRO) for Python classes,
    showing the inheritance hierarchy and all methods defined in each class.

    TYPENAMES  One or more fully qualified type names (e.g. mymodule.MyClass)
    """
    console = Console()
    for typename in typenames:
        # import_class_from_path handles errors via cli_error (exits on failure)
        type_ = import_class_from_path(console, typename)
        if len(typenames) > 1:
            richprint(f"\n{'=' * 60}")
        print_type_hierarchy(type_)


if __name__ == "__main__":  # pragma: no cover
    app()
