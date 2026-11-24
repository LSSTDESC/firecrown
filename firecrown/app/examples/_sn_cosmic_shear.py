"""Supernova and cosmic shear analysis example generator.

Generates an analysis of two independent data sets whose factory returns a list
of likelihoods.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Sequence

import typer

from firecrown.likelihood import NamedParameters

from ..analysis import Model, copy_template, download_from_url
from ..sacc import Transform
from . import _sn_cosmic_shear_template
from ._cosmic_shear import ExampleCosmicShear
from ._sn_srd import ExampleSupernovaSRD, sn_srd_magnitude_parameter


@dataclass
class ExampleSNCosmicShear(ExampleCosmicShear):
    """Supernova and cosmic shear analysis with independent likelihoods.

    The supernova data are the LSST DESC SRD Year 1 sample of
    :class:`ExampleSupernovaSRD`. The cosmic shear data are the synthetic data of
    :class:`ExampleCosmicShear`, configured by the same options. The factory
    returns the two likelihoods as a list, and Firecrown sums their
    log-likelihoods in a :class:`firecrown.likelihood.SumLikelihood`.
    """

    description: ClassVar[str] = (
        "Supernova SRD and synthetic cosmic shear as a list of independent "
        "likelihoods"
    )

    prefix: Annotated[
        str,
        typer.Option(
            help=(
                "Prefix for generated filenames (e.g., 'sn_cosmic_shear' creates "
                "'sn_cosmic_shear.hdf5' and 'sn_cosmic_shear_sn.hdf5')"
            ),
            show_default=True,
        ),
    ] = "sn_cosmic_shear"

    def sn_sacc_path(self, output_path: Path) -> Path:
        """Return the path of the supernova SACC file in the target format.

        :param output_path: Output directory
        :returns: Path ``<prefix>_sn.<sacc_format>`` in output_path
        """
        return output_path / f"{self.prefix}_sn.{self.sacc_format.value}"

    def generate_sacc(self, output_path: Path) -> Path:
        """Download the supernova data and generate the cosmic shear data.

        The supernova file is written to :meth:`sn_sacc_path`, converted to
        ``sacc_format`` if the download has another format. The base class
        converts the returned cosmic shear file.

        :param output_path: Output directory
        :returns: Path to the cosmic shear SACC file
        """
        downloaded = output_path / f"{self.prefix}_sn.sacc"
        sn_sacc = self.sn_sacc_path(output_path)
        download_from_url(
            ExampleSupernovaSRD.data_url,
            downloaded,
            self.console,
            "Downloading SN SRD data...",
        )
        if Transform.detect_format(downloaded) == self.sacc_format:
            downloaded.replace(sn_sacc)
        else:
            Transform(
                sacc_file=downloaded,
                output=sn_sacc,
                output_format=self.sacc_format,
                overwrite=True,
                quiet=True,
            )
            downloaded.unlink()
        return super().generate_sacc(output_path)

    def generate_factory(self, output_path: Path, _sacc: Path) -> Path:
        """Copy the supernova and cosmic shear factory template.

        :param output_path: Output directory
        :param _sacc: SACC file path (unused)
        :returns: Path to factory file
        """
        output_file = output_path / f"{self.prefix}_factory.py"
        copy_template(_sn_cosmic_shear_template, output_file)
        return output_file

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Return both SACC file paths and the number of tomographic bins.

        :param sacc_path: Path to the cosmic shear SACC file
        :returns: Named parameters sn_sacc_file, shear_sacc_file and n_bins
        """
        return NamedParameters(
            {
                "sn_sacc_file": self.get_sacc_file(self.sn_sacc_path(sacc_path.parent)),
                "shear_sacc_file": self.get_sacc_file(sacc_path),
                "n_bins": self.n_bins,
            }
        )

    def get_models(self) -> list[Model]:
        """Define the photo-z shifts and the supernova absolute magnitude.

        :returns: One model with the delta_z parameters and sn_ddf_sample_M
        """
        (shear_model,) = super().get_models()
        return [
            Model(
                name=f"firecrown_{self.prefix}",
                description="Model parameters for supernova and cosmic shear analysis",
                parameters=[*shear_model.parameters, sn_srd_magnitude_parameter()],
            )
        ]

    def get_options_desc(self) -> Sequence[tuple[str, str]]:
        """Return description of the supernova and cosmic shear options."""
        return [
            ("Supernova data", ExampleSupernovaSRD.data_url),
            *super().get_options_desc(),
        ]
