"""This script provides a way to print the class hierarchy of a given type."""

import importlib
import inspect
import sys


def import_type(full_path: str) -> type:
    """Import a type from a full path, returning the type."""
    module_path, type_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, type_name)


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
    print(f"{idx}  {full_type_name(t)}")
    methods = get_defined_methods(t)
    for method in methods:
        print(f"       {method}")


def print_type_hierarchy(top_type: type) -> None:
    """Print the class hierarchy for the given type."""
    print(f"Hierarchy for {full_type_name(top_type)}:")
    for i, t in enumerate(inspect.getmro(top_type)):
        print_one_type(i, t)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_hierarchy.py <typename> [<typename> ...]")
        sys.exit(1)
    typename = sys.argv[1]
    try:
        type_ = import_type(typename)
    except ImportError as e:
        print(f"Could not import type {typename}")
        print("Error message was:\n", e)
        sys.exit(1)

    print_type_hierarchy(type_)
