"""Cluster number counts and weak lensing analysis example generator.

Generates a binned cluster number counts analysis combined with stacked weak
lensing profiles, using a cluster catalog simulated with NumCosmo and NFW
profiles computed with CLMM.
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
from . import _cluster_shear_template
from ._cluster_data import (
    PROFILE_CONCENTRATION,
    PROFILE_Z_INF,
    bin_catalog,
    fiducial_cosmology_spec,
    shear_sacc,
    simulate_catalog,
    stack_profiles,
)

# pylint: disable=duplicate-code

SIMULATED_SURVEY_NAME = "numcosmo_sim_red_richness_shear_dsigma"


class ClusterProfile(StrEnum):
    """Weak lensing profiles available for the cluster shear example."""

    DELTA_SIGMA = "delta_sigma"
    REDUCED_SHEAR = "reduced_shear"


@dataclass
class ExampleClusterShear(AnalysisBuilder):
    """Cluster number counts and stacked weak lensing analysis example.

    Bins simulated clusters in redshift and richness and fits the counts
    together with the stacked excess surface density DeltaSigma or reduced
    tangential shear profile in each bin.
    """

    description: ClassVar[str] = (
        "Cluster number counts with stacked weak lensing profiles"
    )

    prefix: Annotated[
        str,
        typer.Option(
            help=(
                "Prefix for generated filenames (e.g., 'cluster_shear' creates "
                "'cluster_shear.sacc')"
            ),
            show_default=True,
        ),
    ] = "cluster_shear"

    profile: Annotated[
        ClusterProfile,
        typer.Option(
            help="Weak lensing profile to fit",
            show_default=True,
            case_sensitive=False,
        ),
    ] = ClusterProfile.DELTA_SIGMA

    use_cluster_counts: Annotated[
        bool,
        typer.Option(help="Include the cluster counts", show_default=True),
    ] = True

    use_mean_log_mass: Annotated[
        bool,
        typer.Option(help="Include the mean log10 mass", show_default=True),
    ] = False

    seed: Annotated[
        int,
        typer.Option(help="Random seed for the simulated catalog", show_default=True),
    ] = 2

    def generate_sacc(self, output_path: Path) -> Path:
        """Simulate clusters and their lensing profiles in SACC format.

        The SACC file holds counts, mean log10 mass, DeltaSigma and reduced
        shear; the likelihood selects the observables it uses.

        :param output_path: Output directory
        :returns: Path to generated SACC file
        """
        with self.console.status("Simulating cluster catalog..."):
            catalog = simulate_catalog(self.seed, Nc.MultiplicityFuncMassDef.CRITICAL)
            binned = bin_catalog(catalog)
        self.console.print(f"Simulated {len(catalog.z)} clusters")

        with self.console.status("Computing stacked lensing profiles..."):
            profiles = stack_profiles(
                catalog, binned, fiducial_cosmology_spec().to_ccl_cosmology()
            )
            sacc_data = shear_sacc(SIMULATED_SURVEY_NAME, binned, profiles)

        sacc_full_file = output_path / f"{self.prefix}.sacc"
        sacc_data.save_fits(sacc_full_file, overwrite=True)
        return sacc_full_file

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Copy the cluster shear factory template.

        :param output_path: Output directory
        :param _sacc: SACC file path (unused)
        :returns: Path to factory file
        """
        output_file = output_path / f"{self.prefix}_factory.py"
        copy_template(_cluster_shear_template, output_file)
        return output_file

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Return the SACC file, survey name, profile and selected observables.

        :param sacc_path: Path to the SACC data file
        :returns: Named parameters for the likelihood factory
        """
        return NamedParameters(
            {
                "sacc_file": self.get_sacc_file(sacc_path),
                "survey_name": SIMULATED_SURVEY_NAME,
                "profile": self.profile.value,
                "use_cluster_counts": self.use_cluster_counts,
                "use_mean_log_mass": self.use_mean_log_mass,
            }
        )

    def get_models(self) -> list[Model]:
        """Define the mass-richness relation and concentration parameters.

        All parameters are fixed.

        :returns: Model with the Murata mass-richness parameters and the
            cluster concentration
        """
        parameters = [
            ("mass_distribution_mu0", r"\mu_0", 2.7, 3.8, 3.19),
            ("mass_distribution_mu1", r"\mu_1", 0.4, 1.2, 0.868),
            ("mass_distribution_mu2", r"\mu_2", -0.6, -0.02, -0.3),
            ("mass_distribution_sigma0", r"\sigma_0", 0.02, 0.6, 0.33),
            ("mass_distribution_sigma1", r"\sigma_1", -0.05, -0.01, -0.034),
            ("mass_distribution_sigma2", r"\sigma_2", -0.1, 0.1, 0.0),
            (
                "cluster_theory_cluster_concentration",
                "c",
                1.0,
                10.0,
                PROFILE_CONCENTRATION,
            ),
        ]
        return [
            Model(
                name=f"firecrown_{self.prefix}",
                description="Mass-richness relation and halo profile parameters",
                parameters=[
                    Parameter.from_tuple(name, symbol, lower, upper, default, False)
                    for name, symbol, lower, upper, default in parameters
                ],
            )
        ]

    def required_cosmology(self) -> FrameworkCosmology:
        """Return cosmology requirement level."""
        return FrameworkCosmology.LINEAR

    def required_distance_max_z(self) -> float:
        """Return a maximum redshift above the reduced shear z_inf.

        The reduced shear evaluates distances up to z_inf = 10; DeltaSigma
        needs only the default range.
        """
        if self.profile == ClusterProfile.REDUCED_SHEAR:
            return PROFILE_Z_INF * 1.1
        return super().required_distance_max_z()

    def cosmology_analysis_spec(self) -> CCLCosmologySpec:
        """Return the cosmology used to simulate the catalog unless overridden.

        :returns: The cosmology analysis specification
        """
        if self._spec is not None:
            return self._spec
        return fiducial_cosmology_spec()

    def get_options_desc(self) -> Sequence[tuple[str, str]]:
        """Return description of the cluster shear options."""
        return [
            ("Lensing profile", self.profile.value),
            ("Cluster counts", f"{self.use_cluster_counts}"),
            ("Mean log mass", f"{self.use_mean_log_mass}"),
            ("Seed", f"{self.seed}"),
        ]
