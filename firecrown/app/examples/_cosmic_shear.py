"""Cosmic Shear analysis example generator.

This module implements a complete example generator for weak lensing
cosmic shear analysis, creating synthetic galaxy survey data with
realistic noise properties and covariance matrices.
"""

import shutil
from typing import Annotated, ClassVar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sacc
import pyccl as ccl
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from ...utils import upper_triangle_indices
from ._base_example import Example
from . import _cosmic_shear_template
from . import _cosmosis


@dataclass
class ExampleCosmicShear(Example):
    """Generator for Cosmic Shear analysis example.

    Creates a complete example setup for weak lensing cosmic shear analysis,
    including synthetic data in SACC format and corresponding experiment
    configuration files.
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
    ] = 50

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

    def generate_sacc(self, output_path: Path) -> Path:
        """Generate synthetic cosmic shear data in SACC format.

        Creates a complete SACC file containing:
        - Two tomographic redshift bins with Gaussian n(z) distributions
        - Auto and cross-correlation power spectra (C_ell)
        - Realistic noise based on theoretical predictions
        - Diagonal covariance matrix for statistical analysis

        :param output_path: Directory where the SACC file will be created
        """
        sacc_file = f"{self.prefix}.sacc"
        sacc_full_file = output_path / sacc_file

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            # Phase 1: Setup
            task1 = progress.add_task(
                "Setting up cosmology and coordinates...", total=None
            )
            self._show_cosmology_config()
            cosmo = self._create_fiducial_cosmology()
            z_range, ell_range = self._create_coordinate_arrays()
            self._show_coordinate_config(z_range, ell_range)
            progress.update(task1, completed=True)

            # Phase 2: Tracers
            task2 = progress.add_task("Creating tomographic tracers...", total=None)
            np.random.seed(self.seed)
            sacc_data = sacc.Sacc()
            self._show_tracer_config()
            tracers = self._create_tracers(sacc_data, cosmo, z_range)
            progress.update(task2, completed=True)

            # Phase 3: Power spectra
            task3 = progress.add_task("Computing power spectra...", total=None)
            self._show_power_spectrum_config()
            theory_cls = self._generate_power_spectra(
                sacc_data, cosmo, tracers, ell_range
            )
            progress.update(task3, completed=True)

            # Phase 4: Covariance
            task4 = progress.add_task("Adding covariance matrix...", total=None)
            self._show_covariance_config()
            self._add_covariance_matrix(sacc_data, theory_cls)
            progress.update(task4, completed=True)

            # Phase 5: Save
            task5 = progress.add_task("Saving SACC file...", total=None)
            sacc_data.save_fits(sacc_full_file, overwrite=True)
            progress.update(task5, completed=True)

        self.console.print(f"[green]SACC file saved:[/green] {sacc_file}")

        return sacc_full_file

    def _create_fiducial_cosmology(self) -> ccl.Cosmology:
        """Create fiducial cosmology for synthetic data generation.

        Uses standard Planck-like cosmological parameters to generate
        realistic cosmic shear power spectra.

        :return: CCL Cosmology object with fiducial parameters
        """
        return ccl.Cosmology(
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

    def _show_cosmology_config(self) -> None:
        """Display cosmology configuration."""
        config_text = "Omega_c=0.27, Omega_b=0.045, w0=-1.0\nAs=2.1e-9, ns=0.96, h=0.67"
        self.console.print(
            Panel(config_text, title="Fiducial Cosmology", border_style="blue")
        )

    def _show_coordinate_config(
        self, z_range: np.ndarray, ell_range: np.ndarray
    ) -> None:
        """Display coordinate configuration."""
        config_text = (
            f"Redshift: {z_range.min():.2f} - {z_range.max():.2f} ({len(z_range)} points)\n"
            f"Multipoles: {ell_range.min():.0f} - {ell_range.max():.0f} ({len(ell_range)} points)"
        )
        self.console.print(
            Panel(config_text, title="Coordinate Sampling", border_style="cyan")
        )

    def _show_tracer_config(self) -> None:
        """Display tracer configuration."""
        config_text = (
            f"Number of bins: {self.n_bins}\n"
            f"Redshift width (sigma_z): {self.sigma_z}\n"
            f"Random seed: {self.seed}"
        )
        self.console.print(
            Panel(config_text, title="Tomographic Tracers", border_style="green")
        )

    def _show_power_spectrum_config(self) -> None:
        """Display power spectrum configuration."""
        n_correlations = self.n_bins * (self.n_bins + 1) // 2
        config_text = (
            f"Correlations: {n_correlations} (auto + cross)\n"
            f"Noise level: {self.noise_level:.3f}"
        )
        self.console.print(
            Panel(config_text, title="Power Spectra", border_style="yellow")
        )

    def _show_covariance_config(self) -> None:
        """Display covariance configuration."""
        total_points = self.n_ell_points * self.n_bins * (self.n_bins + 1) // 2
        config_text = f"Matrix size: {total_points}x{total_points} (diagonal)"
        self.console.print(
            Panel(config_text, title="Covariance Matrix", border_style="magenta")
        )

    def _create_tracers(
        self, sacc_data: sacc.Sacc, cosmo: ccl.Cosmology, z_range: np.ndarray
    ) -> list[ccl.WeakLensingTracer]:
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
            tracers.append(ccl.WeakLensingTracer(cosmo, dndz=(z_range, nz)))

        self.console.print(
            f"[dim]Created {len(tracers)} tracers with centers at z = {bin_centers}[/dim]"
        )
        return tracers

    def _generate_power_spectra(
        self,
        sacc_data: sacc.Sacc,
        cosmo: ccl.Cosmology,
        tracers: list[ccl.WeakLensingTracer],
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
            # Compute theoretical C_ℓ
            cl_theory = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell_range)

            # Add realistic noise
            noise = np.random.normal(size=len(cl_theory)) * self.noise_level * cl_theory
            cl_noisy = cl_theory + noise

            # Add to SACC data
            sacc_data.add_ell_cl(
                "galaxy_shear_cl_ee", f"trc{i}", f"trc{j}", ell_range, cl_noisy
            )
            theory_cls.append(cl_theory)

        self.console.print(f"[dim]Generated {len(theory_cls)} power spectra[/dim]")
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
        self.console.print(
            f"[dim]Added {covariance.shape[0]}x{covariance.shape[1]} covariance matrix[/dim]"
        )

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Generate example configuration file."""
        template = Path(_cosmic_shear_template.__file__)
        output_file = output_path / f"{self.prefix}_factory.py"
        shutil.copyfile(template, output_file)

        return output_file

    def generate_cosmosis_config(
        self, output_path: Path, sacc_path: Path, factory_path: Path
    ) -> None:
        """Generate CosmoSIS configuration files.

        Creates both the main .ini file and the values .ini file using
        the standardized CosmoSIS utilities.
        """
        cosmosis_ini = output_path / f"cosmosis_{self.prefix}.ini"
        values_ini = output_path / f"cosmosis_{self.prefix}_values.ini"

        # Generate main configuration
        cfg = _cosmosis.create_standard_cosmosis_config(
            prefix=self.prefix,
            factory_filename=factory_path.name,
            sacc_filename=sacc_path.name,
            values_filename=values_ini.name,
            n_bins=self.n_bins,
        )

        # Generate values configuration
        values_cfg = _cosmosis.create_standard_values_config(n_bins=self.n_bins)

        # Write configuration files
        with values_ini.open("w") as fp:
            values_cfg.write(fp)

        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)
