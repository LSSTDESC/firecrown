"""Generate a JSON map of Firecrown symbols to their documentation URLs."""

import inspect
import pkgutil
import json
import re
import sys
from pathlib import Path
import importlib
from types import ModuleType

import typer
from rich.console import Console

# Assuming this script is in firecrown/fctools and the package is 'firecrown'
# Go up two levels to get to the repo root, so 'firecrown' is in the path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

# pylint: disable=wrong-import-position
# Import must occur after sys.path modification to ensure firecrown is found
import firecrown  # noqa: E402

_CONSTANT_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")


def _is_private_name(name: str) -> bool:
    """Check if a name is private (starts with underscore).

    :param name: The name to check
    :return: True if name starts with underscore
    """
    return name.startswith("_")


def _is_excluded_type(obj: object) -> bool:
    """Check if an object is a type that should be handled separately.

    :param obj: The object to check
    :return: True if object is a class, function, or module
    """
    return inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismodule(obj)


def _has_constant_name(name: str) -> bool:
    """Check if a name follows constant naming conventions.

    :param name: The name to check
    :return: True if name matches UPPER_CASE pattern
    """
    return _CONSTANT_NAME_PATTERN.match(name) is not None


def _is_firecrown_instance(obj: object) -> bool:
    """Check if an object is an instance of a Firecrown class.

    :param obj: The object to check
    :return: True if object's class module starts with 'firecrown'
    """
    class_module = obj.__class__.__module__
    return class_module is not None and class_module.startswith("firecrown")


def _is_api_constant(name: str, obj: object) -> bool:
    """Check if an object is a module-level constant that should be documented.

    :param name: The name of the object
    :param obj: The object to check
    :return: True if this is a documentable constant
    """
    return (
        not _is_private_name(name)
        and not _is_excluded_type(obj)
        and (_has_constant_name(name) or _is_firecrown_instance(obj))
    )


def _add_symbol_to_map(
    symbols: dict[str, str],
    modname: str,
    name: str,
    obj: object,
    package_name: str,
) -> None:
    """Add a symbol and its re-exported paths to the symbol map.

    This handles classes, functions, and module-level constants.

    :param symbols: The symbol map to update
    :param modname: The module where the symbol is accessible
    :param name: The name of the symbol
    :param obj: The symbol object
    :param package_name: The root package name
    """
    # Create the public API path
    public_name = f"{modname}.{name}"

    # Handle classes and functions (have __module__ attribute)
    if inspect.isclass(obj) or inspect.isfunction(obj):
        if not obj.__module__.startswith(package_name):
            return

        # Determine the best URL for documentation
        defining_module = obj.__module__
        url = f"api/{defining_module}.html#{defining_module}.{name}"
        symbols[public_name] = url

        # Also add an entry for the defining module path if different
        if obj.__module__ != modname:
            defining_name = f"{obj.__module__}.{name}"
            symbols[defining_name] = url

    # Handle module-level constants
    elif _is_api_constant(name, obj):
        # Constants don't have __module__, so use the current module for URL
        url = f"api/{modname}.html#{modname}.{name}"
        symbols[public_name] = url


def get_all_symbols(package: ModuleType) -> dict[str, str]:
    """Walk through a package and collect modules, classes, functions, and constants.

    This function captures:
    - Classes and functions (with proper handling of re-exports)
    - Module-level constants (following naming conventions)
    - Singleton instances of Firecrown classes

    This ensures that users can reference symbols using the documented public API
    paths rather than needing to know about private implementation modules.

    :param package: The Python package to inspect
    :return: Dictionary mapping symbol names to their documentation URLs
    """
    prefix: str = package.__name__ + "."
    symbols: dict[str, str] = {}

    for _, modname, _ in pkgutil.walk_packages(
        path=package.__path__, prefix=prefix, onerror=lambda x: None
    ):
        module: ModuleType = importlib.import_module(modname)
        # Add the module itself
        symbols[modname] = f"api/{modname}.html"

        for name, obj in inspect.getmembers(module):
            if _is_private_name(name):
                continue

            _add_symbol_to_map(symbols, modname, name, obj, package.__name__)

    return symbols


app = typer.Typer()


@app.command()
def main(
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. If not specified, prints to stdout.",
        file_okay=True,
        dir_okay=False,
    ),
    pretty: bool = typer.Option(
        True,
        "--pretty/--compact",
        help="Pretty-print JSON with indentation (default: pretty).",
    ),
) -> None:
    """Generate a JSON map of Firecrown symbols to their documentation URLs.

    Introspects the Firecrown package to find all classes, functions, and
    module-level constants, then generates a mapping from symbol names to
    their Sphinx documentation URLs.

    By default, outputs to stdout for easy piping. Use --output to write to a file.
    """
    console = Console(stderr=True)  # Status messages to stderr, JSON to stdout

    with console.status("[bold green]Analyzing Firecrown package..."):
        symbols: dict[str, str] = get_all_symbols(firecrown)

    indent = 2 if pretty else None
    json_output = json.dumps(symbols, indent=indent, sort_keys=True)

    if output:
        output.write_text(json_output, encoding="utf-8")
        console.print(
            f"[green]✓[/green] Generated {len(symbols)} symbols → [cyan]{output}[/cyan]"
        )
    else:
        # Print to stdout for piping (use plain print, not console)
        print(json_output)


if __name__ == "__main__":  # pragma: no cover
    app()
