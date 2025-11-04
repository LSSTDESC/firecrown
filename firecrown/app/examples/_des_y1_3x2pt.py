"""DES Y1 3x2pt analysis example generator.

This module implements an example generator for DES Y1 3x2pt analysis,
downloading real data from the DES repository.
"""

import shutil
import urllib.request
from typing import ClassVar, Annotated
from dataclasses import dataclass
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

import typer

from firecrown.likelihood.likelihood import NamedParameters
from ._base_example import Example
from ._types import Model, Parameter
from . import _des_y1_3x2pt_template


@dataclass
class ExampleDESY13x2pt(Example):
    """Generator for DES Y1 3x2pt analysis example.

    Downloads real DES Y1 3x2pt data and creates corresponding
    configuration files for different analysis frameworks.
    """

    description: ClassVar[str] = (
        "DES Y1 3x2pt analysis with cosmic shear, galaxy-galaxy lensing, and galaxy clustering"
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
        """Download DES Y1 3x2pt data file.

        Downloads the SACC file containing DES Y1 3x2pt data from the
        LSST DESC repository.

        :param output_path: Directory where the SACC file will be downloaded
        :return: Path to the downloaded SACC file
        """
        sacc_file = f"{self.prefix}.sacc"
        sacc_full_file = output_path / sacc_file

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Downloading DES Y1 3x2pt data...", total=None)

            try:
                urllib.request.urlretrieve(self.data_url, sacc_full_file)
                progress.update(task, completed=True)
                self.console.print(f"[green]Downloaded:[/green] {sacc_file}")
            except Exception as e:
                self.console.print(f"[red]Download failed:[/red] {e}")
                raise

        return sacc_full_file

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Generate factory file from template."""
        # Copy the factory.py from des_y1_3x2pt examples
        template = Path(_des_y1_3x2pt_template.__file__)
        output_file = output_path / f"{self.prefix}_factory.py"
        shutil.copyfile(template, output_file)
        return output_file

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        if self.use_absolute_path:
            sacc_filename = sacc_path.absolute().as_posix()
        else:
            sacc_filename = sacc_path.name

        return NamedParameters({"sacc_file": sacc_filename})

    def get_models(self) -> list[Model]:
        """Get the models for the example."""
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
                parameters=[
                    Parameter(
                        name=param[0],
                        symbol=param[1],
                        lower_bound=param[2],
                        upper_bound=param[3],
                        scale=param[4],
                        abstol=param[5],
                        default_value=param[6],
                        free=param[7],
                    )
                    for param in parameters
                ],
            )
        ]
