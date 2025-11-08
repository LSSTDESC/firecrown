"""Cosmic shear analysis example generator.

Generates synthetic weak lensing cosmic shear data with realistic
noise and covariance for testing and demonstration purposes.
"""

import shutil
from typing import Annotated, ClassVar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sacc
import pyccl
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from firecrown.likelihood.likelihood import NamedParameters
from ...utils import upper_triangle_indices
from ..analysis import AnalysisBuilder, Model, Parameter
from . import _cosmic_shear_template


@dataclass
class ExampleCosmicShear(AnalysisBuilder):
    """Cosmic shear analysis example with synthetic data.

    Generates synthetic weak lensing data with configurable tomographic bins,
    multipole ranges, and noise levels for testing analysis pipelines.
    """

    description: ClassVar[str] = (
        "Weak lensing cosmic shear analysis with synthetic galaxy data"
    )

    prefix: Annotated[
        str,
        typer.Option(
            help=(
                "Prefix for generated filenames "
                "(e.g., 'cosmic_shear' creates 'cosmic_shear.sacc')"
            ),
            show_default=True,
        ),
    ] = "cosmic_shear"

    seed: Annotated[
        int,
        typer.Option(
            help="Random seed for reproducible synthetic data generation",
            show_default=True,
        ),
    ] = 42

    n_bins: Annotated[
        int,
        typer.Option(
            help="Number of tomographic redshift bins",
            show_default=True,
        ),
    ] = 2

    z_max: Annotated[
        float,
        typer.Option(
            help="Maximum redshift for n(z) distributions",
            show_default=True,
        ),
    ] = 2.0

    n_z_points: Annotated[
        int,
        typer.Option(
            help="Number of redshift points for n(z) sampling",
            show_default=True,
        ),
    ] = 600

    ell_min: Annotated[
        float,
        typer.Option(
            help="Minimum multipole for power spectrum",
            show_default=True,
        ),
    ] = 10.0

    ell_max: Annotated[
        float,
        typer.Option(
            help="Maximum multipole for power spectrum",
            show_default=True,
        ),
    ] = 10000.0

    n_ell_points: Annotated[
        int,
        typer.Option(
            help="Number of multipole points for power spectrum",
            show_default=True,
        ),
    ] = 10

    noise_level: Annotated[
        float,
        typer.Option(
            help="Relative noise level for synthetic data",
            show_default=True,
        ),
    ] = 0.01

    sigma_z: Annotated[
        float,
        typer.Option(
            help="Width of Gaussian redshift distributions",
            show_default=True,
        ),
    ] = 0.25

    def _setup_phase(self, progress, summary, _output_path):
        """Setup phase of SACC generation."""
        task1 = progress.add_task("Setting up cosmology and coordinates...", total=None)
        cosmo = self._create_fiducial_cosmology()
        summary.add_row("[b]Cosmology[/b]", "")
        self._show_cosmology_config(cosmo, summary)

        z_range, ell_range = self._create_coordinate_arrays()
        summary.add_section()
        summary.add_row("[b]Coordinates[/b]", "")
        self._show_coordinate_config(z_range, ell_range, summary)
        progress.update(task1, completed=True)

        return cosmo, z_range, ell_range

    def _tracer_phase(self, progress, summary, cosmo, z_range):
        """Tracer generation phase."""
        task2 = progress.add_task("Creating tomographic tracers...", total=None)
        np.random.seed(self.seed)
        sacc_data = sacc.Sacc()
        bin_centers, tracers = self._create_tracers(sacc_data, cosmo, z_range)

        summary.add_section()
        summary.add_row("[b]Tracers[/b]", "")
        self._show_tracer_config(bin_centers, summary)
        progress.update(task2, completed=True)

        return sacc_data, bin_centers, tracers

    def _spectra_phase(self, progress, summary, sacc_data, cosmo, tracers, ell_range):
        """Power spectra computation phase."""
        task3 = progress.add_task("Computing power spectra...", total=None)
        theory_cls = self._generate_power_spectra(sacc_data, cosmo, tracers, ell_range)
        summary.add_section()
        summary.add_row("[b]Power Spectra[/b]", "")
        self._show_power_spectrum_config(summary)
        progress.update(task3, completed=True)
        return theory_cls

    def _covariance_phase(self, progress, summary, sacc_data, theory_cls):
        """Covariance matrix generation phase."""
        task4 = progress.add_task("Adding covariance matrix...", total=None)
        summary.add_section()
        summary.add_row("[b]Covariance Matrix[/b]", "")
        self._show_covariance_config(summary)
        self._add_covariance_matrix(sacc_data, theory_cls)
        progress.update(task4, completed=True)

    def _save_phase(self, progress, sacc_data, output_path):
        """Save SACC file phase."""
        sacc_full_file = output_path / f"{self.prefix}.sacc"
        task5 = progress.add_task("Saving SACC file...", total=None)
        sacc_data.save_fits(sacc_full_file, overwrite=True)
        progress.update(task5, completed=True)
        return sacc_full_file

    def generate_sacc(self, output_path: Path) -> Path:
        """Generate synthetic cosmic shear data in SACC format.

        Creates SACC file with tomographic bins, power spectra, and covariance.

        :param output_path: Output directory
        :return: Path to generated SACC file
        """
        summary = Table(
            title="Cosmic Shear Example", border_style="blue", show_header=False
        )
        summary.add_column("Parameter", style="cyan", no_wrap=True)
        summary.add_column("Value", style="green")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            # Phase 1: Setup
            cosmo, z_range, ell_range = self._setup_phase(
                progress, summary, output_path
            )

            # Phase 2: Tracers
            sacc_data, _, tracers = self._tracer_phase(
                progress, summary, cosmo, z_range
            )

            # Phase 3: Power spectra
            theory_cls = self._spectra_phase(
                progress, summary, sacc_data, cosmo, tracers, ell_range
            )

            # Phase 4: Covariance
            self._covariance_phase(progress, summary, sacc_data, theory_cls)

            # Phase 5: Save
            sacc_full_file = self._save_phase(progress, sacc_data, output_path)

            progress.console.print(
                f"[dim]Generated {len(theory_cls)} power spectra[/dim]"
            )

        summary.add_section()
        summary.add_row("[b]Output File[/b]", sacc_full_file.as_posix())

        self.console.print(summary)

        return sacc_full_file

    def _create_fiducial_cosmology(self) -> pyccl.Cosmology:
        """Create fiducial cosmology for synthetic data generation.

        Uses standard Planck-like cosmological parameters to generate
        realistic cosmic shear power spectra.

        :return: CCL Cosmology object with fiducial parameters
        """
        return pyccl.Cosmology(
            Omega_c=0.27,
            Omega_b=0.045,
            Omega_k=0.0,
            w0=-1.0,
            wa=0.0,
            A_s=2.1e-9,
            n_s=0.96,
            h=0.67,
        )

    def _create_coordinate_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Create coordinate arrays for redshift and multipole sampling.

        Sets up the redshift range for n(z) distributions and the multipole
        range for power spectrum calculations.

        :return: Tuple of (z_range, ell_range) arrays for sampling
        """
        z_range = np.linspace(0, self.z_max, self.n_z_points) + 0.05
        ell_range = np.logspace(
            np.log10(self.ell_min), np.log10(self.ell_max), self.n_ell_points
        )
        return z_range, ell_range

    def _show_cosmology_config(self, cosmo: pyccl.Cosmology, table: Table) -> None:
        """Display cosmology configuration."""
        params = [
            ("Omega_c", ".3f"),
            ("Omega_b", ".3f"),
            ("w0", ".1f"),
            ("A_s", ".1e"),
            ("n_s", ".2f"),
            ("h", ".2f"),
        ]

        for param, fmt in params:
            table.add_row(param, f"{cosmo[param]:{fmt}}")

    def _show_coordinate_config(
        self, z_range: np.ndarray, ell_range: np.ndarray, table: Table
    ) -> None:
        """Display coordinate configuration."""
        table.add_row("Redshift", f"{z_range.min():.2f} - {z_range.max():.2f}")
        table.add_row("Multipoles", f"{ell_range.min():.0f} - {ell_range.max():.0f}")

    def _show_tracer_config(self, bin_centers: np.ndarray, table: Table) -> None:
        """Display tracer configuration."""
        table.add_row("Number of bins", f"{self.n_bins}")
        table.add_row("Redshift width", f"{self.sigma_z:.2f}")
        table.add_row("Random seed", f"{self.seed}")
        table.add_row("Bin centers", ", ".join(f"{x:.3f}" for x in bin_centers))

    def _show_power_spectrum_config(self, table: Table) -> None:
        """Display power spectrum configuration."""
        n_correlations = self.n_bins * (self.n_bins + 1) // 2

        table.add_row("Correlations", f"{n_correlations} (auto + cross)")
        table.add_row("Noise level", f"{self.noise_level:.3f}")

    def _show_covariance_config(self, table: Table) -> None:
        """Display covariance configuration."""
        total_points = self.n_ell_points * self.n_bins * (self.n_bins + 1) // 2

        table.add_row("Number of points", f"{total_points}")
        table.add_row("Format", "diagonal")

    def _create_tracers(
        self, sacc_data: sacc.Sacc, cosmo: pyccl.Cosmology, z_range: np.ndarray
    ) -> tuple[np.ndarray, list[pyccl.WeakLensingTracer]]:
        """Create tomographic redshift bins and weak lensing tracers.

        Generates two tomographic bins with Gaussian redshift distributions
        centered at different redshifts, representing a typical galaxy survey
        binning scheme.

        :param sacc_data: SACC data object to populate with tracer metadata
        :param cosmo: CCL cosmology object for tracer calculations
        :param z_range: Redshift sampling array
        :return: List of CCL WeakLensingTracer objects for theory calculations
        """
        tracers = []
        # Distribute bin centers evenly across redshift range
        bin_centers = np.linspace(0.2, self.z_max * 0.8, self.n_bins)

        for i, z_mean in enumerate(bin_centers):
            # Gaussian redshift distribution
            nz = np.exp(-0.5 * (z_range - z_mean) ** 2 / self.sigma_z**2)

            # Add tracer to SACC
            tracer_name = f"trc{i}"
            sacc_data.add_tracer("NZ", tracer_name, z_range, nz)

            # Create CCL tracer for theory calculations
            tracers.append(pyccl.WeakLensingTracer(cosmo, dndz=(z_range, nz)))

        return bin_centers, tracers

    def _generate_power_spectra(
        self,
        sacc_data: sacc.Sacc,
        cosmo: pyccl.Cosmology,
        tracers: list[pyccl.WeakLensingTracer],
        ell_range: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate cosmic shear power spectra with realistic noise.

        Computes theoretical C_ell for all auto and cross-correlations between
        tomographic bins, adds Gaussian noise, and stores in SACC format.

        :param sacc_data: SACC data object to populate with measurements
        :param cosmo: CCL cosmology for theoretical predictions
        :param tracers: List of weak lensing tracers for each redshift bin
        :param ell_range: Multipole sampling array
        :return: List of noise-free theoretical power spectra for covariance
        """
        theory_cls = []

        for i, j in upper_triangle_indices(len(tracers)):
            # Compute theoretical C_ell
            cl_theory = pyccl.angular_cl(cosmo, tracers[i], tracers[j], ell_range)

            # Add realistic noise
            noise = np.random.normal(size=len(cl_theory)) * self.noise_level * cl_theory
            cl_noisy = cl_theory + noise

            # Add to SACC data
            sacc_data.add_ell_cl(
                "galaxy_shear_cl_ee", f"trc{i}", f"trc{j}", ell_range, cl_noisy
            )
            theory_cls.append(cl_theory)

        return theory_cls

    def _add_covariance_matrix(
        self, sacc_data: sacc.Sacc, theory_cls: list[np.ndarray]
    ) -> None:
        """Add diagonal covariance matrix based on theoretical predictions.

        Creates a simplified diagonal covariance matrix where uncertainties
        are proportional to the theoretical power spectrum values.

        :param sacc_data: SACC data object to populate with covariance
        :param theory_cls: Theoretical power spectra for uncertainty estimation
        """
        all_theory = np.concatenate(theory_cls)
        cov_diag = (self.noise_level * all_theory) ** 2
        covariance = np.diag(cov_diag)
        sacc_data.add_covariance(covariance)

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Copy cosmic shear factory template.

        :param output_path: Output directory
        :param _sacc: SACC file path (unused)
        :return: Path to factory file
        """
        template = Path(_cosmic_shear_template.__file__)
        output_file = output_path / f"{self.prefix}_factory.py"
        shutil.copyfile(template, output_file)

        return output_file

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        return NamedParameters(
            {"sacc_file": self.get_sacc_file(sacc_path), "n_bins": self.n_bins}
        )

    def get_models(self) -> list[Model]:
        """Define photo-z shift parameters for each tomographic bin.

        :return: Model with delta_z parameters for all bins
        """
        parameters: list[tuple[str, str, float, float, float, float, float, bool]] = [
            (
                f"trc{bin_index}_delta_z",
                rf"\delta_{{z{bin_index}}}",
                -5.0,
                5.0,
                0.05,
                0.0,
                0.5,
                True,
            )
            for bin_index in range(self.n_bins)
        ]
        return [
            Model(
                name=f"firecrown_{self.prefix}",
                description="Model parameters for cosmic shear analysis",
                parameters=[Parameter(*param) for param in parameters],
            )
        ]
