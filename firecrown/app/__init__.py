"""Firecrown Command-Line Interface (CLI).

The `firecrown` command provides tools to inspect, visualize, and manage data,
experiments, and examples used in Firecrown analyses.

Available command groups:
  - `sacc`: Inspect and manipulate SACC data files.
  - `experiment`: Inspect and visualize Firecrown experiment configurations.
  - `examples`: Generate example data and configuration files for different analyses.

Each command group provides additional subcommands.
Use `firecrown <command> --help` for details.
"""

import typer
import firecrown.app.sacc as sacc_app
import firecrown.app.experiment as experiment_app
import firecrown.app.examples as examples_app
import firecrown.app.cosmology as cosmology_app

# Root application ------------------------------------------------------------

MAIN_HELP = """
Firecrown command-line tools for cosmological analyses.


Use 'firecrown COMMAND --help' for detailed help on each command.
"""
app = typer.Typer(no_args_is_help=True, help=MAIN_HELP)

# SACC commands ---------------------------------------------------------------

app_sacc = typer.Typer(no_args_is_help=True)
# - Subcommands
app_sacc.command(
    name="view",
    no_args_is_help=True,
    help="View the contents of a SACC file (e.g., data points, metadata).",
)(sacc_app.View)
app_sacc.command(
    name="convert",
    no_args_is_help=True,
    help="Convert SACC file format.",
)(sacc_app.Convert)

SACC_HELP = """
Inspect, visualize, and convert SACC data files.


Use 'firecrown sacc SUBCOMMAND --help' for detailed help.
"""
app.add_typer(app_sacc, name="sacc", help=SACC_HELP)

# Experiment commands ---------------------------------------------------------
HELP_EXPERIMENT = """
Inspect and visualize Firecrown experiment configurations.


Available subcommands: view


Use 'firecrown experiment SUBCOMMAND --help' for detailed help.
"""
app_experiment = typer.Typer(no_args_is_help=True)
# - Subcommands
app_experiment.command(
    name="view",
    no_args_is_help=True,
    help="Display information about a Firecrown experiment configuration.",
)(experiment_app.View)
app.add_typer(app_experiment, name="experiment", help=HELP_EXPERIMENT)

# Examples commands -----------------------------------------------------------
HELP_EXAMPLES = """
Generate example analyses with synthetic data and configuration files.


Each example creates complete analysis templates for different cosmological analyses.


IMPORTANT: First run 'firecrown examples' to see available examples, then use 'firecrown examples EXAMPLE_NAME --help' for detailed help on a specific example.
"""
app_examples = typer.Typer(no_args_is_help=True)
for example_name, example_cls in examples_app.EXAMPLES_LIST.items():
    app_examples.command(
        name=example_name,
        no_args_is_help=True,
        help=example_cls.description,
    )(example_cls)
app.add_typer(app_examples, name="examples", help=HELP_EXAMPLES)

# Cosmology commands ----------------------------------------------------------
HELP_COSMOLOGY = """
Generate cosmology configuration files for Firecrown analyses.
"""
app.command(name="cosmology", no_args_is_help=True, help=HELP_COSMOLOGY)(
    cosmology_app.Generate
)
