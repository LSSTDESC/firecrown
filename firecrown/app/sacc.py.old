"""SACC data visualization and analysis."""

from typing import Annotated, TypedDict
from abc import ABC, abstractmethod
from enum import Enum
import dataclasses
from pathlib import Path
import sys
import warnings
import io
import re
import contextlib
import typer
import sacc
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from firecrown import metadata_types as mdt
from firecrown import metadata_functions as mdf
from firecrown import data_types as dtype
from firecrown import data_functions as dfunc
from firecrown.likelihood import factories
from . import logging


class OutputHandler(ABC):
    """Abstract base class for handling output from SACC operations.

    Each handler maintains internal state of matched issues and provides
    methods to report them.
    """

    def __init__(self):
        """Initialize the handler with empty state."""
        self._matched_issues: list[str] = []

    def count(self) -> int:
        """Return the number of issues handled by this handler.

        :return: Count of matched issues.
        """
        return len(self._matched_issues)

    @abstractmethod
    def get_title(self) -> str:
        """Get the title for reporting this issue type.

        :return: Title string for console output.
        """

    @abstractmethod
    def get_details(self) -> str | None:
        """Get detailed information about the handled issues.

        :return: Details string for console output, or None if no details.
        """


class MessageHandler(OutputHandler):
    """Handler for complete warning messages."""

    @abstractmethod
    def try_handle(self, message: str) -> bool:
        """Attempt to handle a warning message.

        :param message: The complete warning message to potentially handle.
        :return: True if this handler handled the message, False otherwise.
        """


class StreamHandler(OutputHandler):
    """Handler for line-based output streams (stdout/stderr)."""

    @abstractmethod
    def try_handle(self, lines: list[str]) -> tuple[bool, list[str]]:
        """Attempt to handle lines from a stream.

        :param lines: List of lines from the stream to potentially handle.
        :return: Tuple of (handled, remaining_lines). If handled is True,
            the handler consumed some lines. remaining_lines are the lines
            that were not consumed and should be passed to the next handler.
        """


class TracerNamingViolationHandler(MessageHandler):
    """Handler for SACC convention violation warnings about tracer naming."""

    def __init__(self):
        """Initialize the tracer naming violation handler."""
        super().__init__()
        self._pattern = re.compile(
            (
                r"SACC Convention Violation Detected.*tracer "
                r"'(.*?)'.*tracer '(.*?)'.*data type string '(.*?)'"
            ),
            re.DOTALL | re.IGNORECASE,
        )

    def try_handle(self, message: str) -> bool:
        """Try to handle a tracer naming convention violation warning.

        :param message: The warning message to check.
        :return: True if handled, False otherwise.
        """
        match = self._pattern.search(message)
        if match:
            tracer1, tracer2, data_type = match.groups()
            formatted = (
                f"Tracers '{tracer1}' and '{tracer2}' (data type: '{data_type}')"
            )
            self._matched_issues.append(formatted)
            return True
        return False

    def get_title(self) -> str:
        """Get the title for tracer naming violations.

        :return: Title string with count.
        """
        return f"⚠️  Found {self.count()} tracer naming convention violation(s)"

    def get_details(self) -> str | None:
        """Get details of all tracer naming violations.

        :return: Formatted list of violations.
        """
        if not self._matched_issues:
            return None
        return "\n".join(f"  • {msg}" for msg in sorted(self._matched_issues))


class LegacyCovarianceHandler(MessageHandler):
    """Handler for legacy SACC covariance format warnings."""

    def __init__(self):
        """Initialize the legacy covariance handler."""
        super().__init__()
        self._pattern = re.compile(
            r"older sacc legacy sacc file format.*covariance",
            re.DOTALL | re.IGNORECASE,
        )

    def try_handle(self, message: str) -> bool:
        """Try to handle a legacy covariance format warning.

        :param message: The warning message to check.
        :return: True if handled, False otherwise.
        """
        if self._pattern.search(message):
            self._matched_issues.append("Legacy covariance format detected")
            return True
        return False

    def get_title(self) -> str:
        """Get the title for legacy covariance warnings.

        :return: Title string.
        """
        return "⚠️  Warning: Legacy covariance format detected"

    def get_details(self) -> str | None:
        """Get details for legacy covariance warnings.

        :return: Explanation text.
        """
        if not self._matched_issues:
            return None
        return (
            "  This SACC file uses an older internal format for covariance data.\n"
            "  Consider re-saving the file with a newer SACC version to ensure\n"
            "  compatibility and improved performance."
        )


class MissingSaccOrderingHandler(StreamHandler):
    """Handler for missing sacc_ordering metadata in stdout."""

    def __init__(self):
        """Initialize the missing sacc_ordering handler."""
        super().__init__()
        self._pattern = re.compile(
            r"sacc_ordering.*deprecated", re.IGNORECASE | re.DOTALL
        )

    def try_handle(self, lines: list[str]) -> tuple[bool, list[str]]:
        """Try to handle lines about missing sacc_ordering.

        Looks for multi-line pattern containing 'sacc_ordering' and 'deprecated'.

        :param lines: List of lines from stdout.
        :return: Tuple of (handled, remaining_lines).
        """
        # Join all lines to check for pattern
        remaining_lines = []
        found = False
        for line, next_line in zip(lines, lines[1:]):
            if (
                "Missing sacc_ordering metadata" in line
            ) and "Assuming data rows are in the correct order" in next_line:
                found = True
            else:
                remaining_lines.append(line)
                remaining_lines.append(next_line)
        if found:
            return True, remaining_lines
        return False, lines

    def get_title(self) -> str:
        """Get the title for missing sacc_ordering.

        :return: Title string.
        """
        return "⚠️  Warning: Missing 'sacc_ordering' metadata"

    def get_details(self) -> str | None:
        """Get details for missing sacc_ordering.

        :return: Explanation text.
        """
        if not self._matched_issues:
            return None
        return (
            "  The 'sacc_ordering' column is missing from all data points.\n"
            "  This indicates an older SACC file format (pre-1.0).\n"
            "  Consider re-saving the file with a newer SACC version."
        )


class UnknownWarningHandler(MessageHandler):
    """Catch-all handler for unrecognized warnings."""

    def __init__(self):
        """Initialize the unknown warning handler."""
        super().__init__()
        self._warnings: list[tuple[str, str]] = []  # (category, message)

    def try_handle(self, message: str) -> bool:
        """Always handle any warning (catch-all).

        :param message: The warning message.
        :return: Always True.
        """
        self._warnings.append(("Unknown", message))
        return True

    def count(self) -> int:
        """Return the number of warnings handled.

        :return: Count of warnings.
        """
        return len(self._warnings)

    def get_title(self) -> str:
        """Get the title for unknown warnings.

        :return: Title string with count.
        """
        return f"⚠️  Found {self.count()} unknown warning(s)"

    def get_details(self) -> str | None:
        """Get details of all unknown warnings.

        :return: Formatted list of warnings.
        """
        if not self._warnings:
            return None
        details = []
        for idx, (category, message) in enumerate(self._warnings, 1):
            details.append(f"  Warning {idx}: {category}")
            details.append(f"    {message}")
        return "\n".join(details)


class UnknownStdoutHandler(StreamHandler):
    """Catch-all handler for unrecognized stdout lines."""

    def try_handle(self, lines: list[str]) -> tuple[bool, list[str]]:
        """Always handle all remaining stdout lines (catch-all).

        :param lines: List of lines from stdout.
        :return: Tuple of (True, []) - consumes all lines.
        """
        for line in lines:
            if line.strip():  # Only capture non-empty lines
                self._matched_issues.append(line)
        return True, []

    def get_title(self) -> str:
        """Get the title for unknown stdout.

        :return: Title string.
        """
        return "⚠️  Unknown output from SACC library (stdout):"

    def get_details(self) -> str | None:
        """Get details of all unknown stdout lines.

        :return: Formatted list of lines.
        """
        if not self._matched_issues:
            return None
        return "\n".join(f"  {line}" for line in self._matched_issues)


class UnknownStderrHandler(StreamHandler):
    """Catch-all handler for unrecognized stderr lines."""

    def try_handle(self, lines: list[str]) -> tuple[bool, list[str]]:
        """Always handle all stderr lines (catch-all).

        :param lines: List of lines from stderr.
        :return: Tuple of (True, []) - consumes all lines.
        """
        for line in lines:
            if line.strip():  # Only capture non-empty lines
                self._matched_issues.append(line)
        return True, []

    def get_title(self) -> str:
        """Get the title for unknown stderr.

        :return: Title string.
        """
        return "⚠️  Unknown output from SACC library (stderr):"

    def get_details(self) -> str | None:
        """Get details of all unknown stderr lines.

        :return: Formatted list of lines.
        """
        if not self._matched_issues:
            return None
        return "\n".join(f"  {line}" for line in self._matched_issues)


# Handler types for different output streams
WARNING_HANDLERS: list[type[MessageHandler]] = [
    TracerNamingViolationHandler,
    LegacyCovarianceHandler,
    UnknownWarningHandler,  # Must be last (catch-all)
]

STDOUT_HANDLERS: list[type[StreamHandler]] = [
    MissingSaccOrderingHandler,
    UnknownStdoutHandler,  # Must be last (catch-all)
]

STDERR_HANDLERS: list[type[StreamHandler]] = [
    UnknownStderrHandler,  # Catch-all for stderr
]


QuadOpts = TypedDict(
    "QuadOpts",
    {
        "limit": int,
        "epsabs": float,
        "epsrel": float,
    },
)


def mean_std_tracer(tracer: mdt.InferredGalaxyZDist):
    """Compute the mean and standard deviation of a tracer.

    :param tracer: The galaxy redshift distribution tracer to analyze.
    :return: Tuple of (mean_z, std_z) for the tracer distribution.
    """
    # Create monotonic spline
    spline = PchipInterpolator(tracer.z, tracer.dndz, extrapolate=False)
    quad_opts: QuadOpts = {"limit": 10000, "epsabs": 0.0, "epsrel": 1.0e-3}

    def spline_func(t):
        return spline(t)

    # Normalization
    norm, _ = quad(spline_func, tracer.z[0], tracer.z[-1], **quad_opts)

    # Mean
    mean_z, _ = quad(lambda t: t * spline(t), tracer.z[0], tracer.z[-1], **quad_opts)
    mean_z /= norm

    # Variance
    var_z, _ = quad(
        lambda t: (t - mean_z) ** 2 * spline(t), tracer.z[0], tracer.z[-1], **quad_opts
    )
    std_z = np.sqrt(var_z / norm)

    return mean_z, std_z


@dataclasses.dataclass(kw_only=True)
class Load(logging.Logging):
    """Load and summarize a SACC file."""

    sacc_file: Annotated[
        Path, typer.Argument(help="Path to the SACC file.", show_default=True)
    ]
    allow_mixed_types: Annotated[
        bool,
        typer.Option(
            "--allow-mixed-types",
            help=(
                "Allow measurements with types from different sets "
                "(e.g., galaxy source + lens types)."
            ),
        ),
    ] = False

    def __post_init__(self) -> None:
        """Load and display metadata from the SACC file."""
        super().__post_init__()
        self._load_sacc_file()

    def _load_sacc_file(self) -> None:
        """Load the SACC file, with error handling for missing or unreadable files."""
        self.console.rule("[bold blue]Loading SACC file[/bold blue]")
        self.console.print(f"[cyan]File:[/cyan] {self.sacc_file}")
        self.console.print(f"[cyan]Allow mixed types:[/cyan] {self.allow_mixed_types}")
        try:
            if not self.sacc_file.exists():
                raise typer.BadParameter(f"SACC file not found: {self.sacc_file}")
            self.sacc_data = factories.load_sacc_data(self.sacc_file.as_posix())
        except Exception as e:
            self.console.print(f"[bold red]Failed to load SACC file:[/bold red] {e}")
            raise


@dataclasses.dataclass(kw_only=True)
class View(Load):
    """Display a summary of the SACC file."""

    plot_covariance: Annotated[
        bool, typer.Option(help="Plot the covariance matrix.", show_default=True)
    ] = False
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help=(
                "Validate SACC file: naming convention compliance, tracer ordering, "
                "metadata attributes, and other quality checks."
            ),
        ),
    ] = False

    def __post_init__(self) -> None:
        """Display a summary of the SACC file."""
        super().__post_init__()
        self._show_sacc_summary()
        self._show_tracers()
        self._show_harmonic_bins()
        self._show_real_bins()
        self._show_final_summary()
        if self.check:
            self._check_sacc_quality()
        if self.plot_covariance:
            self._plot_covariance()

    def _show_sacc_summary(self) -> None:
        """Show a summary of the SACC file."""
        n_tracers = len(self.sacc_data.tracers)
        n_data_points = len(self.sacc_data.mean)
        if self.sacc_data.covariance is None:
            n_cov_elements = 0
        else:
            n_cov_elements = self.sacc_data.covariance.dense.shape[0]
        self.all_tracers = mdf.extract_all_tracers_inferred_galaxy_zdists(
            self.sacc_data, self.allow_mixed_types
        )
        self.all_tracers.sort(key=lambda t: t.bin_name)
        self.console.rule("[bold blue]SACC Summary[/bold blue]")
        self.console.print(
            Panel.fit(
                f"[bold]Number of tracers:[/bold] {n_tracers}\n"
                f"[bold]Data points:[/bold] {n_data_points}\n"
                f"[bold]Covariance elements:[/bold] {n_cov_elements}",
                title="SACC Summary",
                border_style="green",
            )
        )

    def _show_tracers(self) -> None:
        """Show the tracers in the SACC file."""
        if len(self.all_tracers) > 0:
            self.console.rule("[bold magenta]Tracers[/bold magenta]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Name")
            table.add_column("TypeSource")
            table.add_column("z min-max (density)")
            table.add_column("dndz mean, std")
            table.add_column("Measurements")
            for tracer in self.all_tracers:
                measurements_str = ", ".join(
                    [m.name for m in sorted(tracer.measurements)]
                )
                z_str = f"{tracer.z.min():5.3f}-{tracer.z.max():5.3f}"
                mean, std = mean_std_tracer(tracer)
                dndz_str = f"{mean:5.3f} +/- {std:5.3f}"
                table.add_row(
                    tracer.bin_name,
                    tracer.type_source,
                    z_str,
                    dndz_str,
                    f"{{{measurements_str}}}",
                )
            self.console.print(table)

    def _show_harmonic_bins(self) -> None:
        """Show the harmonic bins in the SACC file."""
        self.bin_comb_harmonic: list[dtype.TwoPointMeasurement] = (
            dfunc.extract_all_harmonic_data(self.sacc_data, self.allow_mixed_types)
        )
        self.bin_dict_harmonic = {
            (
                b.metadata.XY.x.bin_name,
                b.metadata.XY.x_measurement,
                b.metadata.XY.y.bin_name,
                b.metadata.XY.y_measurement,
            ): b
            for b in self.bin_comb_harmonic
        }
        if len(self.bin_comb_harmonic) > 0:
            self.console.rule(
                "[bold magenta]Bin combinations [Harmonic][/bold magenta]"
            )
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("x-bin")
            table.add_column("y-bin")
            table.add_column("x-meas")
            table.add_column("y-meas")
            table.add_column("ells: min-max (length, mean-delta)")
            table.add_column("Window?")
            table.add_column("SACC Datatype")
            for m in self.bin_comb_harmonic:
                assert isinstance(m.metadata, mdt.TwoPointHarmonic)
                bin_harmonic: mdt.TwoPointHarmonic = m.metadata
                window_str = (
                    Text("Yes", style="green")
                    if bin_harmonic.window is not None
                    else Text("No", style="red")
                )
                ells_min = min(bin_harmonic.ells)
                ells_max = max(bin_harmonic.ells)
                ells_n = len(bin_harmonic.ells)
                ells_mean_delta = (ells_max - ells_min) / (ells_n - 1)
                ells_str = (
                    f"{ells_min:6} - {ells_max:6} "
                    f"({ells_n:6}, {ells_mean_delta:.2f})"
                )
                table.add_row(
                    bin_harmonic.XY.x.bin_name,
                    bin_harmonic.XY.y.bin_name,
                    bin_harmonic.XY.x_measurement.name,
                    bin_harmonic.XY.y_measurement.name,
                    ells_str,
                    window_str,
                    bin_harmonic.get_sacc_name(),
                )
            self.console.print(table)

    def _show_real_bins(self) -> None:
        """Show the real bins in the SACC file."""
        self.bin_comb_real: list[dtype.TwoPointMeasurement] = (
            dfunc.extract_all_real_data(self.sacc_data)
        )
        self.bin_dict_real = {
            (
                b.metadata.XY.x.bin_name,
                b.metadata.XY.x_measurement,
                b.metadata.XY.y.bin_name,
                b.metadata.XY.y_measurement,
            ): b
            for b in self.bin_comb_real
        }
        if len(self.bin_comb_real) > 0:
            self.console.rule("[bold magenta]Bin combinations [Real][/bold magenta]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("x-bin")
            table.add_column("y-bin")
            table.add_column("x-meas")
            table.add_column("y-meas")
            table.add_column("thetas: min-max (length, mean-delta)")
            table.add_column("SACC Datatype")
            for m in self.bin_comb_real:
                assert isinstance(m.metadata, mdt.TwoPointReal)
                bin_real: mdt.TwoPointReal = m.metadata
                thetas_min = min(bin_real.thetas)
                thetas_max = max(bin_real.thetas)
                thetas_n = len(bin_real.thetas)
                thetas_mean_delta = (thetas_max - thetas_min) / (thetas_n - 1)
                thetas_str = (
                    f"{thetas_min:6.2f} - {thetas_max:6.2f} "
                    f"({thetas_n:6}, {thetas_mean_delta:.2f})"
                )
                table.add_row(
                    bin_real.XY.x.bin_name,
                    bin_real.XY.y.bin_name,
                    bin_real.XY.x_measurement.name,
                    bin_real.XY.y_measurement.name,
                    thetas_str,
                    bin_real.get_sacc_name(),
                )
            self.console.print(table)

    def _show_final_summary(self) -> None:
        self.console.rule("[bold green]Summary[/bold green]")
        self.console.print(
            f"[yellow]Total harmonic bins:[/yellow] {len(self.bin_comb_harmonic)}"
        )
        self.console.print(
            f"[yellow]Total real bins:[/yellow] {len(self.bin_comb_real)}"
        )
        self.console.print(f"[yellow]Total tracers:[/yellow] {len(self.all_tracers)}")

    def _check_sacc_quality(self) -> None:
        """Validate SACC file quality and compliance."""
        self.console.rule("[bold blue]SACC Quality Checks[/bold blue]")

        # Step 1: Capture all output and warnings during SACC operations
        stdout_buffer, stderr_buffer, captured_warnings, validation_error = (
            self._capture_sacc_operations()
        )

        # Step 2: Create handler instances for each stream
        warning_handlers: list[MessageHandler] = [
            handler_cls() for handler_cls in WARNING_HANDLERS
        ]
        stdout_handlers: list[StreamHandler] = [
            handler_cls() for handler_cls in STDOUT_HANDLERS
        ]
        stderr_handlers: list[StreamHandler] = [
            handler_cls() for handler_cls in STDERR_HANDLERS
        ]

        # Step 3: Process warnings through warning handlers
        for warning_msg in captured_warnings:
            message_str = str(warning_msg.message)
            for msg_handler in warning_handlers:
                if msg_handler.try_handle(message_str):
                    break

        # Step 4: Process stdout lines through stdout handlers
        stdout_lines = stdout_buffer.splitlines()
        for stream_handler in stdout_handlers:
            if stdout_lines:
                handled, stdout_lines = stream_handler.try_handle(stdout_lines)
                if handled and not stdout_lines:
                    break  # All lines consumed

        # Step 5: Process stderr lines through stderr handlers
        stderr_lines = stderr_buffer.splitlines()
        for stream_handler in stderr_handlers:
            if stderr_lines:
                handled, stderr_lines = stream_handler.try_handle(stderr_lines)
                if handled and not stderr_lines:
                    break  # All lines consumed

        # Step 6: Collect all handlers and check for issues
        all_handlers: list[OutputHandler] = []
        all_handlers.extend(warning_handlers)
        all_handlers.extend(stdout_handlers)
        all_handlers.extend(stderr_handlers)
        total_issues = sum(h.count() for h in all_handlers)
        has_validation_error = validation_error is not None

        if total_issues == 0 and not has_validation_error:
            self.console.print("[bold green]✅ All quality checks passed![/bold green]")
            return

        # Step 7: Report validation error if present
        if has_validation_error:
            self.console.print(
                f"[bold red]❌ Validation Error:[/bold red] {validation_error}"
            )
            self.console.print()

        # Step 8: Report all issues from handlers
        for handler in all_handlers:
            if handler.count() > 0:
                self.console.print(f"[yellow]{handler.get_title()}[/yellow]")
                details = handler.get_details()
                if details:
                    self.console.print(details)
                self.console.print()

    def _capture_sacc_operations(
        self,
    ) -> tuple[str, str, list[warnings.WarningMessage], str | None]:
        """Capture stdout, stderr, and warnings from SACC operations.

        :return: Tuple of (stdout_content, stderr_content, warnings_list,
            validation_error).
        """
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        captured_warnings: list[warnings.WarningMessage] = []
        validation_error: str | None = None

        with (
            contextlib.redirect_stdout(stdout_buffer),
            contextlib.redirect_stderr(stderr_buffer),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            try:
                # Load SACC data
                sacc_data_raw = sacc.Sacc.load_fits(str(self.sacc_file))

                # Extract tracers (this may trigger warnings about naming conventions)
                _ = mdf.extract_all_tracers_inferred_galaxy_zdists(
                    sacc_data_raw, allow_mixed_types=False
                )
                captured_warnings = list(w)
            except ValueError as e:
                validation_error = str(e)

        return (
            stdout_buffer.getvalue(),
            stderr_buffer.getvalue(),
            captured_warnings,
            validation_error,
        )

    def _plot_covariance(self) -> None:
        """Plot the covariance matrix with annotations for harmonic and real bins."""
        if self.sacc_data.covariance is None:
            raise typer.BadParameter(
                f"No covariance found in SACC file: {self.sacc_file}"
            )

        all_bins = self.bin_comb_harmonic + self.bin_comb_real

        cor_ordered = self._get_ordered_correlation(all_bins)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = self._plot_correlation_matrix(ax, cor_ordered)
        self._add_bin_annotations(ax, all_bins)
        self._add_plot_decorations(fig, ax, im)
        plt.show()

    def _get_ordered_correlation(self, all_bins):
        """Get the ordered correlation matrix."""
        assert self.sacc_data.covariance is not None
        cov = self.sacc_data.covariance.dense
        cor = np.corrcoef(cov)
        indices_ordered = np.concatenate([b.indices for b in all_bins])
        return cor[np.ix_(indices_ordered, indices_ordered)]

    def _plot_correlation_matrix(self, ax, cor_ordered):
        """Plot the correlation matrix."""
        return ax.matshow(
            cor_ordered,
            cmap="RdBu_r",  # Diverging colormap for positive/negative values
            norm=Normalize(vmin=-1, vmax=1),
        )

    def _add_bin_annotations(self, ax, all_bins):
        """Add bin annotations to the plot."""
        legend_patches = []
        current_index = 0
        for i, b in enumerate(all_bins):
            length = len(b.indices)
            start = current_index - 0.5
            end = current_index + length - 0.5

            color = f"C{i % 10}"  # cycle through default Matplotlib colors
            ax.axvspan(start, end, color=color, alpha=0.08, zorder=2)
            ax.axhspan(start, end, color=color, alpha=0.08, zorder=2)

            legend_patches.append(
                Patch(facecolor=color, alpha=0.3, label=str(b.metadata))
            )

            current_index += length
        ax.legend(handles=legend_patches, title="Bins", loc="best", fontsize=8)

    def _add_plot_decorations(self, fig, ax, im):
        """Add title and colorbar to the plot."""
        fig.colorbar(im, ax=ax, label="Correlation")

        ax.set_title("Correlation Matrix")
        plt.tight_layout()


class SaccFormat(str, Enum):
    """Enum for SACC file formats."""

    FITS = "fits"
    HDF5 = "hdf5"


@dataclasses.dataclass(kw_only=True)
class Convert(logging.Logging):
    """Convert SACC files between FITS and HDF5 formats."""

    input_file: Annotated[
        Path, typer.Argument(help="Input SACC file path", show_default=True)
    ]
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output file path. If not specified, "
                "uses input filename with new extension."
            ),
        ),
    ] = None
    input_format: Annotated[
        SaccFormat | None,
        typer.Option(
            "--input-format",
            "-f",
            case_sensitive=False,
            help="Force input format (overrides automatic detection).",
        ),
    ] = None
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite output file if it exists.")
    ] = False

    def __post_init__(self) -> None:
        """Convert the SACC file."""
        super().__post_init__()

        if not self.input_file.exists():
            self.console.print(
                f"[bold red]ERROR: Input file not found: {self.input_file}[/bold red]"
            )
            sys.exit(1)

        self._convert_file()

    def _detect_format(self, filepath: Path) -> str:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()

        if suffix == ".fits":
            return "fits"
        if suffix in (".hdf5", ".h5"):
            return "hdf5"
        raise ValueError(
            f"Cannot detect format from extension '{suffix}'. "
            "Use --input-format to specify."
        )

    def _determine_output_path(
        self, input_path: Path, output: Path | None, target_format: str
    ) -> Path:
        """Determine the output file path."""
        if output:
            return output

        stem = input_path.stem
        if target_format == "fits":
            return input_path.parent / f"{stem}.fits"
        return input_path.parent / f"{stem}.hdf5"

    def _convert_file(self) -> None:
        """Perform the file conversion."""
        # Detect or use specified input format
        if self.input_format:
            src_format = self.input_format.value
            self.console.print(
                f"Using specified input format: [bold]{src_format.upper()}[/bold]"
            )
        else:
            try:
                src_format = self._detect_format(self.input_file)
                self.console.print(
                    f"Detected input format: [bold]{src_format.upper()}[/bold]"
                )
            except ValueError as e:
                self.console.print(f"[bold red]ERROR: {e}[/bold red]")
                sys.exit(1)

        # Determine target format
        target_format = "hdf5" if src_format == "fits" else "fits"

        # Determine output path
        output_path = self._determine_output_path(
            self.input_file, self.output, target_format
        )

        # Check if output exists
        if output_path.exists() and not self.overwrite:
            self.console.print(
                f"[bold red]ERROR: Output file '{output_path}' already exists. "
                "Use --overwrite to replace it.[/bold red]"
            )
            sys.exit(1)

        # Read and convert
        self._read_and_convert_file(src_format, output_path, target_format)

        # Display success info
        self._display_conversion_summary(src_format, output_path, target_format)

    def _read_and_convert_file(
        self, src_format: str, output_path: Path, target_format: str
    ) -> None:
        """Read input file and write to output format."""
        # Read input file
        self.console.print(
            f"Reading {src_format.upper()} file: [cyan]{self.input_file}[/cyan]"
        )
        try:
            if src_format == "fits":
                data = sacc.Sacc.load_fits(str(self.input_file))
            else:  # hdf5
                data = sacc.Sacc.load_hdf5(str(self.input_file))
        except OSError:
            self.console.print(
                "[bold red]ERROR: Failed to read input file as SACC data.[/bold red]"
            )
            self.console.print(
                f"The file may not be a valid SACC {src_format.upper()} file.",
            )
            sys.exit(1)

        # Write output file
        self.console.print(
            f"Writing {target_format.upper()} file: [cyan]{output_path}[/cyan]"
        )
        try:
            if target_format == "fits":
                data.save_fits(str(output_path), overwrite=self.overwrite)
            else:  # hdf5
                if self.overwrite and output_path.exists():
                    output_path.unlink()
                data.save_hdf5(str(output_path))
        except OSError as e:
            self.console.print(
                f"[bold red]ERROR: Failed to write SACC data to output file: "
                f"{e}[/bold red]"
            )
            sys.exit(1)

    def _display_conversion_summary(
        self, src_format: str, output_path: Path, target_format: str
    ) -> None:
        """Display conversion summary with file sizes."""
        input_size = self.input_file.stat().st_size
        output_size = output_path.stat().st_size

        self.console.print("\n" + "=" * 60)
        self.console.print("✅ [bold green]Conversion successful![/bold green]")
        self.console.print("=" * 60)
        input_info = f"({src_format.upper()}, {input_size:,} bytes)"
        self.console.print(f"Input:  [cyan]{self.input_file}[/cyan] {input_info}")
        self.console.print(
            f"Output: [cyan]{output_path}[/cyan] "
            f"({target_format.upper()}, {output_size:,} bytes)"
        )

        if output_size < input_size:
            ratio = (1 - output_size / input_size) * 100
            self.console.print(f"Size reduction: [bold green]{ratio:.1f}%[/bold green]")
        elif output_size > input_size:
            ratio = (output_size / input_size - 1) * 100
            self.console.print(f"Size increase: [bold red]{ratio:.1f}%[/bold red]")
        else:
            self.console.print("Size unchanged")
