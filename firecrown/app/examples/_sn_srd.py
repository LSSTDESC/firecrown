"""Supernova SRD analysis example generator.

Generates a complete supernova analysis example using LSST DESC SRD Year 1 data.
"""

from typing import ClassVar, Annotated
from dataclasses import dataclass
from pathlib import Path

import typer

from firecrown.likelihood.likelihood import NamedParameters
from ..analysis import (
    AnalysisBuilder,
    Model,
    Parameter,
    FrameworkCosmology,
    PriorUniform,
    CCLCosmologyParameters,
    CCLCosmologyPriors,
    CCLCosmologyAnalysisSpec,
    download_from_url,
    copy_template,
)
from . import _sn_srd_template


@dataclass
class ExampleSupernovaSRD(AnalysisBuilder):
    """Supernova SRD analysis example.

    Downloads LSST DESC SRD Year 1 supernova data and generates
    framework configurations for distance modulus fitting.
    """

    description: ClassVar[str] = "Supernova SRD analysis with LSST DESC synthetic data"

    data_url: ClassVar[str] = (
        "https://github.com/LSSTDESC/"
        "firecrown/releases/download/files-v1.0.0/srd-y1.sacc"
    )

    prefix: Annotated[
        str,
        typer.Option(
            help=(
                "Prefix for generated filenames (e.g., 'sn_srd' creates 'sn_srd.sacc')"
            ),
            show_default=True,
        ),
    ] = "sn_srd"

    def generate_sacc(self, output_path: Path) -> Path:
        """Download supernova SRD data from LSST DESC repository.

        :param output_path: Output directory
        :return: Path to downloaded SACC file
        """
        sacc_full_file = output_path / f"{self.prefix}.sacc"
        download_from_url(
            self.data_url, sacc_full_file, self.console, "Downloading SN SRD data..."
        )
        return sacc_full_file

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Copy supernova factory template.

        :param output_path: Output directory
        :param _sacc: SACC file path (unused)
        :return: Path to factory file
        """
        output_file = output_path / f"{self.prefix}_factory.py"
        copy_template(_sn_srd_template, output_file)
        return output_file

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Return SACC file path for likelihood construction.

        :return: Named parameters with sacc_file path
        """
        return NamedParameters({"sacc_file": self.get_sacc_file(sacc_path)})

    def get_models(self) -> list[Model]:
        """Define supernova absolute magnitude parameter.

        :return: Single model with M parameter
        """
        return [
            Model(
                name=f"firecrown_{self.prefix}",
                description="Model for SN SRD example",
                parameters=[
                    Parameter(
                        name="sn_ddf_sample_M",
                        symbol=r"\mathcal{M}",
                        lower_bound=-20.0,
                        upper_bound=-19.0,
                        scale=0.1,
                        abstol=0.0,
                        default_value=-19.3,
                        free=True,
                    )
                ],
            )
        ]

    def required_cosmology(self):
        """Return cosmology requirement level."""
        return FrameworkCosmology.BACKGROUND

    def cosmology_analysis_spec(self) -> CCLCosmologyAnalysisSpec:
        """Return the cosmology analysis specification.

        :return: The cosmology analysis specification
        """
        cosmology = CCLCosmologyParameters(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.67,
            n_s=0.96,
            A_s=2.1e-9,
        )
        priors = CCLCosmologyPriors(
            Omega_c=PriorUniform(lower=0.06, upper=0.46),
        )
        return CCLCosmologyAnalysisSpec(cosmology=cosmology, priors=priors)
