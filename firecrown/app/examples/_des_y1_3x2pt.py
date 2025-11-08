"""DES Y1 3x2pt analysis example generator.

Generates a complete DES Y1 3x2pt analysis example with cosmic shear,
galaxy-galaxy lensing, and galaxy clustering.
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
    download_from_url,
    copy_template,
)
from . import _des_y1_3x2pt_template


@dataclass
class ExampleDESY13x2pt(AnalysisBuilder):
    """DES Y1 3x2pt analysis example.

    Downloads real DES Y1 3x2pt data (cosmic shear, galaxy-galaxy lensing,
    galaxy clustering) and generates framework configurations.
    """

    description: ClassVar[str] = (
        "DES Y1 3x2pt analysis with cosmic shear, galaxy-galaxy lensing, "
        "and galaxy clustering"
    )

    data_url: ClassVar[str] = (
        "https://github.com/LSSTDESC/"
        "firecrown/releases/download/files-v1.0.0/des_y1_3x2pt.sacc"
    )

    prefix: Annotated[
        str,
        typer.Option(
            help=(
                "Prefix for generated filenames "
                "(e.g., 'des_y1_3x2pt' creates 'des_y1_3x2pt.sacc')"
            ),
            show_default=True,
        ),
    ] = "des_y1_3x2pt"

    def generate_sacc(self, output_path: Path) -> Path:
        """Download DES Y1 3x2pt data from LSST DESC repository.

        :param output_path: Output directory
        :return: Path to downloaded SACC file
        """
        sacc_full_file = output_path / f"{self.prefix}.sacc"
        download_from_url(
            self.data_url,
            sacc_full_file,
            self.console,
            "Downloading DES Y1 3x2pt data...",
        )
        return sacc_full_file

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Copy DES Y1 3x2pt factory template.

        :param output_path: Output directory
        :param _sacc: SACC file path (unused)
        :return: Path to factory file
        """
        output_file = output_path / f"{self.prefix}_factory.py"
        copy_template(_des_y1_3x2pt_template, output_file)
        return output_file

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Return SACC file path for likelihood construction.

        :param sacc_path: Path to the SACC data file
        :return: Named parameters with sacc_data path
        """
        return NamedParameters({"sacc_data": self.get_sacc_file(sacc_path)})

    def get_models(self) -> list[Model]:
        """Define DES Y1 systematic and bias parameters.

        :return: Model with IA, photo-z, bias, and multiplicative bias parameters
        """
        parameters: list[tuple[str, str, float, float, float, float, float, bool]] = [
            ("ia_bias", r"\beta_{\mathrm{ia}}", -5.0, 5.0, 0.05, 0.0, 0.5, True),
            ("alphaz", r"\alpha_z", -5.0, 5.0, 0.05, 0.0, 0.0, True),
            ("z_piv", r"z_{\mathrm{pivot}}", 0.0, 1.0, 0.05, 0.0, 0.62, False),
            ("lens0_bias", r"b_0", 0.8, 3.0, 0.05, 0.0, 1.4, True),
            ("lens1_bias", r"b_1", 0.8, 3.0, 0.05, 0.0, 1.6, True),
            ("lens2_bias", r"b_2", 0.8, 3.0, 0.05, 0.0, 1.6, True),
            ("lens3_bias", r"b_3", 0.8, 3.0, 0.05, 0.0, 1.9, True),
            ("lens4_bias", r"b_4", 0.8, 3.0, 0.05, 0.0, 2.0, True),
            ("lens0_delta_z", r"\delta z_0", -0.05, 0.05, 0.05, 0.0, 0.001, True),
            ("lens1_delta_z", r"\delta z_1", -0.05, 0.05, 0.05, 0.0, 0.002, True),
            ("lens2_delta_z", r"\delta z_2", -0.05, 0.05, 0.05, 0.0, 0.001, True),
            ("lens3_delta_z", r"\delta z_3", -0.05, 0.05, 0.05, 0.0, 0.003, True),
            ("lens4_delta_z", r"\delta z_4", -0.05, 0.05, 0.05, 0.0, 0.000, True),
            ("src0_mult_bias", r"m_0", 0.001, 0.012, 0.05, 0.0, 0.012, True),
            ("src1_mult_bias", r"m_1", 0.001, 0.012, 0.05, 0.0, 0.012, True),
            ("src2_mult_bias", r"m_2", 0.001, 0.012, 0.05, 0.0, 0.012, True),
            ("src3_mult_bias", r"m_3", 0.001, 0.012, 0.05, 0.0, 0.012, True),
            ("src0_delta_z", r"\delta z_0", -0.05, 0.05, 0.005, 0.0, -0.001, True),
            ("src1_delta_z", r"\delta z_1", -0.05, 0.05, 0.005, 0.0, -0.019, True),
            ("src2_delta_z", r"\delta z_2", -0.05, 0.05, 0.005, 0.0, 0.009, True),
            ("src3_delta_z", r"\delta z_3", -0.05, 0.05, 0.005, 0.0, -0.018, True),
        ]
        return [
            Model(
                name=f"firecrown_{self.prefix}",
                description="DES Y1 3x2pt parameters",
                parameters=[Parameter(*param) for param in parameters],
            )
        ]
