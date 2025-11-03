"""SN SRD analysis example generator.

This module implements an example generator for supernova SRD analysis,
downloading real data from the LSST DESC repository.
"""

import shutil
import urllib.request
from typing import ClassVar, Annotated
from dataclasses import dataclass
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn
from numcosmo_py.helper import register_model_class
from numcosmo_py import Ncm

import typer

from firecrown.likelihood.likelihood import NamedParameters
from ._base_example import Example
from . import _sn_srd_template
from . import _cosmosis
from . import _cobaya
from . import _numcosmo


@dataclass
class ExampleSupernovaSRD(Example):
    """Generator for SN SRD analysis example.

    Downloads real supernova data from LSST DESC repository and creates corresponding
    configuration files for different analysis frameworks.
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
                "(e.g., 'cosmic_shear' creates 'cosmic_shear.sacc')"
            ),
            show_default=True,
        ),
    ] = "sn_srd"

    def generate_sacc(self, output_path: Path) -> Path:
        """Download SN SRD data file.

        Downloads the SACC file containing supernova data from the
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
            task = progress.add_task("Downloading SN SRD data...", total=None)

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
        template = Path(_sn_srd_template.__file__)
        output_file = output_path / f"{self.prefix}_factory.py"
        shutil.copyfile(template, output_file)
        return output_file

    def generate_cosmosis_config(
        self, output_path: Path, sacc_path: Path, factory_path: Path
    ) -> None:
        """Generate CosmoSIS configuration files for the SN SRD example.

        Produces a main CosmoSIS ini and a values ini using the standard helpers.
        """
        cosmosis_ini = output_path / f"cosmosis_{self.prefix}.ini"
        values_ini = output_path / f"cosmosis_{self.prefix}_values.ini"
        model_name = f"firecrown_{self.prefix}"

        cfg = _cosmosis.create_standard_cosmosis_config(
            prefix=self.prefix,
            factory_path=factory_path,
            sacc_path=sacc_path,
            values_path=values_ini,
            output_path=output_path,
            model_list=[model_name],
            use_absolute_path=self.use_absolute_path,
        )

        # Add example-specific comments for the SN SRD example
        _cosmosis.add_comment_block(
            cfg,
            "firecrown_likelihood",
            "Supernova SRD example: the factory below will construct "
            "a supernova likelihood.\n"
            "Provide `sacc_file` as a build parameter (it is already "
            "filled by this example).",
        )

        # Write values file and main ini
        values_cfg = _cosmosis.create_standard_values_config()

        # Firecrown-specific parameters for two-point analysis
        values_cfg.add_section(model_name)
        _cosmosis.add_comment_block(
            values_cfg,
            model_name,
            "Supernova SRD example: the sample magnitude\n",
        )
        values_cfg.set(model_name, "sn_ddf_sample_m", "-20.0 -19.3 -19.0")

        with values_ini.open("w") as fp:
            values_cfg.write(fp)

        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)

    def generate_cobaya_config(
        self, output_path: Path, sacc_path: Path, factory_path: Path
    ) -> None:
        """Generate a Cobaya YAML configuration for the SN SRD example.

        Uses the standard Cobaya helper to produce a minimal test/run configuration
        that points to the generated factory and SACC file.
        """
        cobaya_yaml = output_path / f"cobaya_{self.prefix}.yaml"

        cfg = _cobaya.create_standard_cobaya_config(
            factory_path=factory_path,
            sacc_path=sacc_path,
            likelihood_name="firecrown_likelihood",
            use_absolute_path=self.use_absolute_path,
        )

        cfg["params"]["sn_ddf_sample_M"] = {
            "ref": -19.3,
            "prior": {"min": -20.0, "max": -19.0},
        }

        _cobaya.write_cobaya_config(cfg, cobaya_yaml)

    def generate_numcosmo_config(
        self, output_path: Path, sacc_path: Path, factory_path: Path
    ) -> None:
        """Generate NumCosmo configuration files for the SN SRD example.

        Creates a NumCosmo YAML experiment file and a builders file using the
        shared helper in `_numcosmo` and the NumCosmo `Serialize` utility.
        """
        # Initialize NumCosmo runtime
        Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

        model_name = f"firecrown_{self.prefix}"
        model_list = [model_name]

        build_parameters = NamedParameters({})

        config = _numcosmo.create_standard_numcosmo_config(
            factory_path=factory_path,
            sacc_path=sacc_path,
            build_parameters=build_parameters,
            model_list=model_list,
            use_absolute_path=self.use_absolute_path,
        )

        model_builder = Ncm.ModelBuilder.new(
            Ncm.Model, model_name, "Model for cosmic shear"
        )
        model_builder.add_sparam(
            r"\mathcal{M}",
            "sn_ddf_sample_M",
            -20.0,
            -19.0,
            0.01,
            0.0,
            -19.3,
            Ncm.ParamType.FREE,
        )
        FirecrownModel = register_model_class(model_builder)
        mset = config.get("model-set")
        assert isinstance(mset, Ncm.MSet)
        mset.set(FirecrownModel())

        numcosmo_yaml = output_path / f"numcosmo_{self.prefix}.yaml"
        builders_file = numcosmo_yaml.with_suffix(".builders.yaml")

        # Create an empty model builders container (the helper may not require
        # a populated builders list for simple examples)
        model_builders = Ncm.ObjDictStr.new()  # pylint: disable=no-value-for-parameter
        model_builders.add(model_name, model_builder)

        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        ser.dict_str_to_yaml_file(config, numcosmo_yaml.as_posix())
        ser.dict_str_to_yaml_file(model_builders, builders_file.as_posix())
