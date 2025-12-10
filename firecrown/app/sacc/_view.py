"""View command for SACC files."""

import contextlib
import dataclasses
import io
import warnings
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from firecrown import data_functions as dfunc
from firecrown import data_types as dtype
from firecrown import metadata_functions as mdf
from firecrown import metadata_types as mdt
from firecrown.likelihood import factories

from ._handlers import (
    MessageHandler,
    OutputHandler,
    STDERR_HANDLERS,
    STDOUT_HANDLERS,
    StreamHandler,
    WARNING_HANDLERS,
)
from ._load import Load
from ._utils import mean_std_tracer


@dataclasses.dataclass(kw_only=True)
class View(Load):
    """Display comprehensive summary of SACC file contents and run quality checks.

    This command provides detailed inspection of SACC files including:

    - Tracers and their properties (redshift distributions, measurement types)
    - Harmonic and real-space bin combinations
    - Covariance matrix visualization (optional)
    - Quality checks for naming convention compliance (with --check flag)

    The quality checks validate:

    - SACC naming convention compliance (tracer order matches data type string)
    - Tracer ordering according to canonical ordering (CMB < Clusters < Galaxies)
    - Metadata attributes consistency

    CLI Usage::

        # Basic inspection
        firecrown sacc view data.sacc

        # Run quality checks
        firecrown sacc view data.sacc --check

        # View with covariance plot
        firecrown sacc view data.sacc --plot-covariance

        # Full inspection with checks and plot
        firecrown sacc view data.sacc --check --plot-covariance

    Python Usage::

        from pathlib import Path
        from firecrown.app.sacc import View

        # Basic view
        View(sacc_file=Path("data.sacc"))

        # With quality checks
        View(sacc_file=Path("data.sacc"), check=True)

    :param sacc_file: Path to the SACC file (inherited from Load)
    :type sacc_file: Path
    :param plot_covariance: Plot the covariance matrix with bin annotations.
    :type plot_covariance: bool
    :param check: Validate SACC file for naming convention compliance, tracer ordering,
        and other quality issues.
    :type check: bool
    :param allow_mixed_types: Allow tracers with mixed measurement types
        (inherited from Load).
    :type allow_mixed_types: bool

    .. seealso::
        :class:`Transform` - For fixing detected issues

        ``docs/sacc_usage.rst`` - Comprehensive guide to SACC conventions
    """

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
        """Validate SACC file quality and compliance with naming conventions.

        This method runs comprehensive quality checks on the SACC file:

        1. **Naming Convention Compliance**: Validates that tracer order matches
           the order of measurement types in data type strings according to SACC
           naming conventions.

        2. **Tracer Ordering**: Checks that tracers follow canonical ordering
           (CMB < Clusters < Galaxies) and detects reversed tracer pairs.

        3. **Metadata Validation**: Verifies that all required metadata attributes
           are present and consistent.

        The method uses a handler-based architecture to process warnings and output:

        - Warning handlers detect deprecation warnings about convention violations
        - Stream handlers parse stdout/stderr for quality issues
        - All handlers report findings in a structured format

        Results are printed to console with:

        - Green checkmark if all checks pass
        - Yellow warnings for detected issues with detailed descriptions
        - Red errors for validation failures
        - Recommendations for fixing issues using CLI tools

        Common Issues Detected:

        - Tracers in wrong order for cross-correlation measurements
        - Mixed measurement types within a single tracer
        - Missing or inconsistent metadata

        :returns: None
        :rtype: None

        .. rubric:: Example Output

        All checks passing::

            ═══════════════════ SACC Quality Checks ════════════════════
            ✅ All quality checks passed!

        With issues detected::

            ═══════════════════ SACC Quality Checks ════════════════════
            ❌ Tracer Ordering Issues Detected

            The following data types have tracers in the wrong order:
            - For data type galaxy_shearDensity_cl_e:
              Expected order: (src0, lens0)
              Actual order:   (lens0, src0)

            Recommendation: Use 'firecrown sacc transform --fix-ordering'

        .. seealso::
            :meth:`Transform._fix_ordering` - Method to automatically fix ordering
                issues

            ``docs/sacc_usage.rst`` - Detailed explanation of SACC conventions
        """
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

        # Report quality check results
        self._report_quality_check_results(
            warning_handlers, stdout_handlers, stderr_handlers, validation_error
        )

    def _report_quality_check_results(
        self,
        warning_handlers: list[MessageHandler],
        stdout_handlers: list[StreamHandler],
        stderr_handlers: list[StreamHandler],
        validation_error: str | None,
    ) -> None:
        """Report quality check results to console.

        This method collects all handlers and reports any issues found during
        quality checks. Results are printed with appropriate formatting:

        - Green checkmark if all checks pass
        - Yellow warnings for detected issues with detailed descriptions
        - Red errors for validation failures

        :param warning_handlers: List of warning message handlers
        :type warning_handlers: list[MessageHandler]
        :param stdout_handlers: List of stdout stream handlers
        :type stdout_handlers: list[StreamHandler]
        :param stderr_handlers: List of stderr stream handlers
        :type stderr_handlers: list[StreamHandler]
        :param validation_error: Error message if validation failed, None otherwise
        :type validation_error: str | None
        :returns: None
        :rtype: None

        .. note::
            This is Step 6-8 of the quality check pipeline. It collects all
            handlers, counts issues, and reports findings to console.
        """
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

        This method loads the SACC file and extracts tracers in a controlled
        environment that captures all output streams and warnings. This allows
        detailed quality checking without polluting the main output.

        The method performs:

        1. Loading SACC data using Firecrown factories
        2. Extracting tracers (which may trigger naming convention warnings)
        3. Capturing all warnings (especially DeprecationWarnings)
        4. Catching validation errors for non-compliant files

        :returns: Tuple containing:
            - stdout_content: Captured standard output as string
            - stderr_content: Captured standard error as string
            - warnings_list: List of warning messages captured during operations
            - validation_error: Error message if validation failed, None otherwise
        :rtype: tuple[str, str, list[warnings.WarningMessage], str | None]

        .. note::
            This method uses context managers to safely redirect streams and
            ensure cleanup even if exceptions occur.
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
                sacc_data_raw = factories.load_sacc_data(str(self.sacc_file))

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
        assert (
            self.sacc_data.covariance is not None
        ), "Covariance matrix is not available."
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
