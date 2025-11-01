"""Experiment visualization and analysis."""

from typing import Annotated
import dataclasses
from pathlib import Path
import typer
from rich.table import Table
from rich.panel import Panel

from firecrown.likelihood import factories
from . import logging


@dataclasses.dataclass
class Load(logging.Logging):
    """Experiment data visualization and analysis."""

    experiment_file: Annotated[
        Path, typer.Argument(help="Path to the experiment file.", show_default=True)
    ]

    def __post_init__(self) -> None:
        """Loads experiment."""
        super().__post_init__()
        self._load_experiment()

    def _load_experiment(self) -> None:
        """Load the experiment file, with error handling for missing or unreadable files."""
        self.console.rule("[bold blue]Loading SACC file[/bold blue]")
        self.console.print(f"[cyan]File:[/cyan] {self.experiment_file}")
        try:
            if not self.experiment_file.exists():
                raise typer.BadParameter(
                    f"Experiment file not found: {self.experiment_file}"
                )
            self.tp_experiment = factories.TwoPointExperiment.load_from_yaml(
                self.experiment_file
            )
        except Exception as e:
            self.console.print(
                f"[bold red]Failed to load the experiment file:[/bold red] {e}"
            )
            raise


@dataclasses.dataclass
class View(Load):
    """Display a summary of the experiment file."""

    def __post_init__(self) -> None:
        """Loads experiment."""
        super().__post_init__()
        self._print_factories()

    def _print_factories(self) -> None:
        """Print a summary table of the factories configured in the experiment."""
        tp_factory = self.tp_experiment.two_point_factory

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Factory Type")
        table.add_column("TypeSource")
        table.add_column("Per-bin systematics")
        table.add_column("Global systematics")

        def fmt_sys(seq):
            if not seq:
                return "-"
            return ", ".join(getattr(s, "type", str(s)) for s in seq)

        # Weak lensing factories
        for wl in tp_factory.weak_lensing_factories:
            table.add_row(
                "WeakLensing",
                str(wl.type_source),
                fmt_sys(wl.per_bin_systematics),
                fmt_sys(wl.global_systematics),
            )

        # Number counts factories
        for nc in tp_factory.number_counts_factories:
            table.add_row(
                "NumberCounts",
                str(nc.type_source),
                fmt_sys(nc.per_bin_systematics),
                fmt_sys(nc.global_systematics),
            )

        # CMB factories
        for cmb in tp_factory.cmb_factories:
            table.add_row(
                "CMBConvergence", str(cmb.type_source), fmt_sys([]), fmt_sys([])
            )

        # Print data source info
        table2 = Table(show_header=False)
        table2.add_row(
            "Data source:", str(self.tp_experiment.data_source.sacc_data_file)
        )
        if self.tp_experiment.data_source.filters is not None:
            table2.add_row("Filters:", str(self.tp_experiment.data_source.filters))

        self.console.print(Panel(table, title="Factories"))
        self.console.print(Panel(table2, title="Data Source"))
