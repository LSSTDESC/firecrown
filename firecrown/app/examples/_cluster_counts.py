"""Cluster number counts analysis example generator.

Generates a binned cluster number counts and mean mass analysis, using either
a cluster catalog simulated with NumCosmo or the SDSS redMaPPer measurements
of Costanzi et al. (2019).
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, Sequence

import typer
from numcosmo_py import Nc

from firecrown.likelihood import NamedParameters

from ..analysis import (
    AnalysisBuilder,
    CCLCosmologySpec,
    FrameworkCosmology,
    Model,
    Parameter,
    copy_template,
)
from . import _cluster_counts_template
from ._cluster_data import (
    SDSS_SURVEY_NAME,
    bin_catalog,
    counts_sacc,
    fiducial_cosmology_spec,
    sdss_sacc,
    simulate_catalog,
)

SIMULATED_SURVEY_NAME = "numcosmo_simulated_redshift_richness"


class ClusterDataset(StrEnum):
    """Data sets available for the cluster number counts example."""

    NUMCOSMO = "numcosmo"
    SDSS = "sdss"


@dataclass
class ExampleClusterCounts(AnalysisBuilder):
    """Cluster number counts and mean mass analysis example.

    Bins clusters in redshift and richness and fits the counts and/or the
    mean log10 mass in each bin, modeling the mass-richness relation with the
    Murata et al. (2019) parametrization.
    """

    description: ClassVar[str] = (
        "Cluster number counts and mean mass binned in redshift and richness"
    )

    prefix: Annotated[
        str,
        typer.Option(
            help=(
                "Prefix for generated filenames (e.g., 'cluster_counts' creates "
                "'cluster_counts.sacc')"
            ),
            show_default=True,
        ),
    ] = "cluster_counts"

    dataset: Annotated[
        ClusterDataset,
        typer.Option(
            help=(
                "Data set: a catalog simulated with NumCosmo, or the SDSS "
                "redMaPPer counts and masses of Costanzi et al. (2019)"
            ),
            show_default=True,
            case_sensitive=False,
        ),
    ] = ClusterDataset.NUMCOSMO

    use_cluster_counts: Annotated[
        bool,
        typer.Option(help="Include the cluster counts", show_default=True),
    ] = True

    use_mean_log_mass: Annotated[
        bool,
        typer.Option(help="Include the mean log10 mass", show_default=True),
    ] = True

    seed: Annotated[
        int,
        typer.Option(
            help="Random seed for the simulated catalog (numcosmo data set only)",
            show_default=True,
        ),
    ] = 32

    def __post_init__(self) -> None:
        """Validate the selected observables and run the generation workflow."""
        if not (self.use_cluster_counts or self.use_mean_log_mass):
            raise ValueError(
                "At least one of use_cluster_counts and use_mean_log_mass "
                "must be enabled."
            )
        super().__post_init__()

    def generate_sacc(self, output_path: Path) -> Path:
        """Create the cluster counts and mean mass data in SACC format.

        :param output_path: Output directory
        :returns: Path to generated SACC file
        """
        if self.dataset == ClusterDataset.SDSS:
            sacc_data = sdss_sacc()
        else:
            with self.console.status("Simulating cluster catalog..."):
                catalog = simulate_catalog(self.seed, Nc.MultiplicityFuncMassDef.MEAN)
                sacc_data = counts_sacc(SIMULATED_SURVEY_NAME, bin_catalog(catalog))
            self.console.print(f"Simulated {len(catalog.z)} clusters")

        sacc_full_file = output_path / f"{self.prefix}.sacc"
        sacc_data.save_fits(sacc_full_file, overwrite=True)
        return sacc_full_file

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Copy the cluster counts factory template.

        :param output_path: Output directory
        :param _sacc: SACC file path (unused)
        :returns: Path to factory file
        """
        output_file = output_path / f"{self.prefix}_factory.py"
        copy_template(_cluster_counts_template, output_file)
        return output_file

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Return the SACC file, survey name and selected observables.

        :param sacc_path: Path to the SACC data file
        :returns: Named parameters for the likelihood factory
        """
        return NamedParameters(
            {
                "sacc_file": self.get_sacc_file(sacc_path),
                "survey_name": self._survey_name(),
                "use_cluster_counts": self.use_cluster_counts,
                "use_mean_log_mass": self.use_mean_log_mass,
            }
        )

    def get_models(self) -> list[Model]:
        """Define the free mass-richness relation parameters.

        :returns: Model with the Murata mass-richness parameters
        """
        parameters = [
            ("mass_distribution_mu0", r"\mu_0", 2.7, 3.8, 3.1),
            ("mass_distribution_mu1", r"\mu_1", 0.4, 1.2, 0.8),
            ("mass_distribution_mu2", r"\mu_2", -0.6, -0.02, -0.3),
            ("mass_distribution_sigma0", r"\sigma_0", 0.02, 0.6, 0.3),
            ("mass_distribution_sigma1", r"\sigma_1", -0.05, -0.01, -0.03),
            ("mass_distribution_sigma2", r"\sigma_2", -0.1, 0.1, 0.06),
        ]
        return [
            Model(
                name=f"firecrown_{self.prefix}",
                description="Mass-richness relation parameters",
                parameters=[
                    Parameter.from_tuple(name, symbol, lower, upper, default, True)
                    for name, symbol, lower, upper, default in parameters
                ],
            )
        ]

    def required_cosmology(self) -> FrameworkCosmology:
        """Return cosmology requirement level."""
        return FrameworkCosmology.LINEAR

    def cosmology_analysis_spec(self) -> CCLCosmologySpec:
        """Return the cosmology used to simulate the catalog unless overridden.

        :returns: The cosmology analysis specification
        """
        if self._spec is not None:
            return self._spec
        return fiducial_cosmology_spec()

    def get_options_desc(self) -> Sequence[tuple[str, str]]:
        """Return description of the cluster counts options."""
        options = [
            ("Data set", self.dataset.value),
            ("Cluster counts", f"{self.use_cluster_counts}"),
            ("Mean log mass", f"{self.use_mean_log_mass}"),
        ]
        if self.dataset == ClusterDataset.NUMCOSMO:
            options.append(("Seed", f"{self.seed}"))
        return options

    def _survey_name(self) -> str:
        """Return the name of the survey tracer in the SACC file."""
        return {
            ClusterDataset.NUMCOSMO: SIMULATED_SURVEY_NAME,
            ClusterDataset.SDSS: SDSS_SURVEY_NAME,
        }[self.dataset]
