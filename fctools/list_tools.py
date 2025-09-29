#!/usr/bin/env python
"""List all available fctools and their descriptions."""

import click


TOOLS = {
    "coverage_to_tsv.py": "Convert pytest-cov JSON coverage data to TSV format",
    "coverage_summary.py": "Analyze test coverage output in JSON format",
    "print_hierarchy.py": "Print class hierarchy (MRO) for Python types",
    "print_code.py": "Display class definitions with attributes and decorators",
    "tracer.py": "Trace execution of Python scripts or modules",
}


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
    """
    click.echo("Available fctools:\n")

    for tool, description in TOOLS.items():
        if verbose:
            click.echo(f"  {tool}")
            click.echo(f"    {description}")
            click.echo(f"    Usage: python fctools/{tool} --help")
            click.echo()
        else:
            click.echo(f"  {tool:<20} - {description}")

    if not verbose:
        click.echo("\nUse --verbose for detailed information about each tool.")
        click.echo("Use 'python fctools/TOOL.py --help' for tool-specific help.")


if __name__ == "__main__":
    main()
