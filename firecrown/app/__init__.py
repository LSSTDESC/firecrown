"""Firecrown Command-Line Interface (CLI).

The `firecrown` command provides tools to inspect, visualize, and manage data,
experiments, and examples used in Firecrown analyses.

Available command groups:
  - `sacc`: Inspect and manipulate SACC data files.
  - `experiment`: Inspect and visualize Firecrown experiment configurations.
  - `examples`: List and retrieve example configurations and data files.

Each command group provides additional subcommands.
Use `firecrown <command> --help` for details.
"""

import typer
import firecrown.app.sacc as sacc_app
import firecrown.app.experiment as experiment_app
import firecrown.app.examples as examples_app

# Root application ------------------------------------------------------------

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Firecrown main application.\n\n"
        "Use this command-line interface to inspect data, view experiments, "
        "and manage example configurations. "
        "For help on a specific topic, run:\n\n"
        "  firecrown <command> --help"
    ),
)

# SACC commands ---------------------------------------------------------------

app_sacc = typer.Typer(
    no_args_is_help=True,
    help=(
        "Inspect, visualize, and manipulate SACC data files.\n\n"
        "Examples:\n"
        "  firecrown sacc view my_data.sacc"
    ),
)
app.add_typer(app_sacc, name="sacc", help="Work with SACC data files.")

# Experiment commands ---------------------------------------------------------

app_experiment = typer.Typer(
    no_args_is_help=True,
    help=(
        "Inspect and visualize Firecrown experiment configurations.\n\n"
        "Examples:\n"
        "  firecrown experiment view my_experiment.yml"
    ),
)
app.add_typer(app_experiment, name="experiment", help="Work with experiments.")

# Examples commands -----------------------------------------------------------

app_examples = typer.Typer(
    no_args_is_help=True,
    help=(
        "List and retrieve Firecrown example configurations.\n\n"
        "Examples:\n"
        "  firecrown examples list\n"
        "  firecrown examples fetch two_point_basic"
    ),
)
app.add_typer(app_examples, name="examples", help="Access bundled examples.")

# Subcommands -----------------------------------------------------------------

app_sacc.command(
    name="view",
    no_args_is_help=True,
    help="View the contents of a SACC file (e.g., data points, metadata).",
)(sacc_app.View)

app_experiment.command(
    name="view",
    no_args_is_help=True,
    help="Display information about a Firecrown experiment configuration.",
)(experiment_app.View)

app_examples.command(
    name="list",
    no_args_is_help=True,
    help="List available example configurations shipped with Firecrown.",
)(examples_app.List)
