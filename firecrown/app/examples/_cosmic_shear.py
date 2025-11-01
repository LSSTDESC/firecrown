"""Cosmic Shear analysis example generator.

This module implements a complete example generator for weak lensing
cosmic shear analysis, creating synthetic galaxy survey data with
realistic noise properties and covariance matrices.
"""

from typing import Annotated
from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

import numpy as np
import sacc
import pyccl as ccl
import typer

from ...utils import upper_triangle_indices
from ._base_example import Example


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

    files_prefix: Annotated[
        str,
        typer.Option(
            help="Prefix for generated filenames (e.g., 'cosmic_shear' creates 'cosmic_shear.sacc')",
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

    def generate_sacc(self, output_path: Path) -> None:
        """Generate synthetic cosmic shear data in SACC format.

        Creates a complete SACC file containing:
        - Two tomographic redshift bins with Gaussian n(z) distributions
        - Auto and cross-correlation power spectra (C_ℓ)
        - Realistic noise based on theoretical predictions
        - Diagonal covariance matrix for statistical analysis

        :param output_path: Directory where the SACC file will be created
        """

        sacc_file = f"{self.files_prefix}.sacc"
        sacc_full_file = output_path / sacc_file

        self.console.print("[yellow]Creating fiducial cosmology...[/yellow]")
        cosmo = self._create_fiducial_cosmology()
        z_range, ell_range = self._create_coordinate_arrays()

        np.random.seed(self.seed)
        sacc_data = sacc.Sacc()

        self.console.print("[yellow]Generating tomographic tracers...[/yellow]")
        tracers = self._create_tracers(sacc_data, cosmo, z_range)

        self.console.print("[yellow]Computing power spectra...[/yellow]")
        theory_cls = self._generate_power_spectra(sacc_data, cosmo, tracers, ell_range)

        self.console.print("[yellow]Adding covariance matrix...[/yellow]")
        self._add_covariance_matrix(sacc_data, theory_cls)

        sacc_data.save_fits(sacc_full_file, overwrite=True)
        self.console.print(f"[green]✓ SACC file saved:[/green] {sacc_file}")

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
        z_range = np.linspace(0, 2, 50) + 0.05
        ell_range = np.logspace(1, 4, 10)
        return z_range, ell_range

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
        bin_centers = [0.25, 0.75]
        sigma_z = 0.25

        for i, z_mean in enumerate(bin_centers):
            # Gaussian redshift distribution
            nz = np.exp(-0.5 * (z_range - z_mean) ** 2 / sigma_z**2)

            # Add tracer to SACC
            tracer_name = f"trc{i}"
            sacc_data.add_tracer("NZ", tracer_name, z_range, nz)

            # Create CCL tracer for theory calculations
            tracers.append(ccl.WeakLensingTracer(cosmo, dndz=(z_range, nz)))

        return tracers

    def _generate_power_spectra(
        self,
        sacc_data: sacc.Sacc,
        cosmo: ccl.Cosmology,
        tracers: list[ccl.WeakLensingTracer],
        ell_range: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate cosmic shear power spectra with realistic noise.

        Computes theoretical C_ℓ for all auto and cross-correlations between
        tomographic bins, adds Gaussian noise, and stores in SACC format.

        :param sacc_data: SACC data object to populate with measurements
        :param cosmo: CCL cosmology for theoretical predictions
        :param tracers: List of weak lensing tracers for each redshift bin
        :param ell_range: Multipole sampling array
        :return: List of noise-free theoretical power spectra for covariance
        """
        theory_cls = []
        noise_level = 0.01

        for i, j in upper_triangle_indices(len(tracers)):
            # Compute theoretical C_ℓ
            cl_theory = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell_range)

            # Add realistic noise
            noise = np.random.normal(size=len(cl_theory)) * noise_level * cl_theory
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
        noise_level = 0.01
        all_theory = np.concatenate(theory_cls)
        cov_diag = (noise_level * all_theory) ** 2
        covariance = np.diag(cov_diag)
        sacc_data.add_covariance(covariance)
