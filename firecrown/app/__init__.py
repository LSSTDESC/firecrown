"""Firecrown application.


Firecrown main application entry point.
"""

import typer
import firecrown.app.sacc as sacc_app

app = typer.Typer(no_args_is_help=True, help="Firecrown main application.")
app_view = typer.Typer(
    no_args_is_help=True,
    help="Inspect and visualize SACC data and experiment configurations.",
)

app.add_typer(app_view, name="view", help="Inspect and visualize data.")

app_view.command(name="sacc", no_args_is_help=True, help="View SACC data.")(
    sacc_app.LoadSACC
)
