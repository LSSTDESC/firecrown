"""Supernova SRD analysis example generator.

Generates a complete supernova analysis example using LSST DESC SRD Year 1 data.
"""

import shutil
import urllib.request
from typing import ClassVar, Annotated
from dataclasses import dataclass
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

import typer

from firecrown.likelihood.likelihood import NamedParameters
from ..analysis import AnalysisBuilder, Model, Parameter
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
                "Prefix for generated filenames "
                "(e.g., 'sn_srd' creates 'sn_srd.sacc')"
            ),
            show_default=True,
        ),
    ] = "sn_srd"

    def generate_sacc(self, output_path: Path) -> Path:
        """Download supernova SRD data from LSST DESC repository.

        :param output_path: Output directory
        :return: Path to downloaded SACC file
        """
        sacc_file = f"{self.prefix}.sacc"
        sacc_full_file = output_path / sacc_file

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Downloading SN SRD data...", total=None)

            try:
                urllib.request.urlretrieve(self.data_url, sacc_full_file)
                progress.update(task, completed=True)
                self.console.print(f"[dim]Downloaded from {self.data_url}[/dim]")
            except Exception as e:
                self.console.print(f"[red]Download failed:[/red] {e}")
                raise

        return sacc_full_file

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Copy supernova factory template.

        :param output_path: Output directory
        :param _sacc: SACC file path (unused)
        :return: Path to factory file
        """
        template = Path(_sn_srd_template.__file__)
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
