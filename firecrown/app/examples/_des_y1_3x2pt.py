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
from numcosmo_py.helper import register_model_class
from numcosmo_py import Ncm

import typer

from firecrown.likelihood.likelihood import NamedParameters
from ._base_example import Example
from . import _des_y1_3x2pt_template
from . import _cosmosis
from . import _cobaya
from . import _numcosmo


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

    def generate_cosmosis_config(
        self, output_path: Path, sacc_path: Path, factory_path: Path
    ) -> None:
        """Generate CosmoSIS configuration files."""
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

        values_cfg = _cosmosis.create_standard_values_config()

        values_cfg.add_section(model_name)
        _cosmosis.add_comment_block(
            values_cfg,
            model_name,
            "DES Y1 3x2pt parameters",
        )

        values_cfg.set(model_name, "ia_bias", "-5.0 0.5 5.0")
        values_cfg.set(model_name, "alphaz", "-5.0 0.0 5.0")
        values_cfg.set(model_name, "z_piv", "0.62")

        for i in range(5):
            values_cfg.set(model_name, f"lens{i}_bias", f"0.8 {1.4 + i*0.2:.1f} 3.0")

        for i in range(4):
            values_cfg.set(model_name, f"src{i}_delta_z", "-0.05 0.0 0.05")

        for i in range(5):
            values_cfg.set(model_name, f"lens{i}_delta_z", "-0.05 0.0 0.05")

        for i in range(4):
            values_cfg.set(model_name, f"src{i}_mult_bias", "0.001 0.012 0.1")

        with values_ini.open("w") as fp:
            values_cfg.write(fp)

        with cosmosis_ini.open("w") as fp:
            cfg.write(fp)

    def generate_cobaya_config(
        self, output_path: Path, sacc_path: Path, factory_path: Path
    ) -> None:
        """Generate Cobaya configuration files."""
        cobaya_yaml = output_path / f"cobaya_{self.prefix}.yaml"

        cfg = _cobaya.create_standard_cobaya_config(
            factory_path=factory_path,
            sacc_path=sacc_path,
            use_absolute_path=self.use_absolute_path,
            likelihood_name="firecrown_likelihood",
        )

        cfg["params"]["ia_bias"] = {"ref": 0.5, "prior": {"min": -5.0, "max": 5.0}}
        cfg["params"]["alphaz"] = {"ref": 0.0, "prior": {"min": -5.0, "max": 5.0}}
        cfg["params"]["z_piv"] = 0.62

        for i in range(5):
            cfg["params"][f"lens{i}_bias"] = {
                "ref": 1.4 + i * 0.2,
                "prior": {"min": 0.8, "max": 3.0},
            }

        for i in range(4):
            cfg["params"][f"src{i}_delta_z"] = {
                "ref": 0.0,
                "prior": {"min": -0.05, "max": 0.05},
            }

        for i in range(5):
            cfg["params"][f"lens{i}_delta_z"] = {
                "ref": 0.0,
                "prior": {"min": -0.05, "max": 0.05},
            }

        for i in range(4):
            cfg["params"][f"src{i}_mult_bias"] = {
                "ref": 0.012,
                "prior": {"min": 0.001, "max": 0.1},
            }

        _cobaya.write_cobaya_config(cfg, cobaya_yaml)

    def generate_numcosmo_config(
        self, output_path: Path, sacc_path: Path, factory_path: Path
    ) -> None:
        """Generate NumCosmo configuration files."""
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
            Ncm.Model, model_name, "Model for DES Y1 3x2pt"
        )

        model_builder.add_sparam(
            "ia_\\mathrm{bias}", "ia_bias", -5.0, 5.0, 0.1, 0.0, 0.5, Ncm.ParamType.FREE
        )
        model_builder.add_sparam(
            "\\alpha_z", "alphaz", -5.0, 5.0, 0.1, 0.0, 0.0, Ncm.ParamType.FREE
        )
        model_builder.add_sparam(
            "z_\\mathrm{piv}",
            "z_piv",
            0.496,
            0.744,
            0.0062,
            0.0,
            0.62,
            Ncm.ParamType.FIXED,
        )

        for i in range(5):
            model_builder.add_sparam(
                f"b_{{\\mathrm{{lens{i}}}}}",
                f"lens{i}_bias",
                0.8,
                3.0,
                0.022,
                0.0,
                1.4 + i * 0.2,
                Ncm.ParamType.FREE,
            )

        for i in range(4):
            model_builder.add_sparam(
                f"\\delta z_{{\\mathrm{{src{i}}}}}",
                f"src{i}_delta_z",
                -0.05,
                0.05,
                0.001,
                0.0,
                0.0,
                Ncm.ParamType.FREE,
            )

        for i in range(5):
            model_builder.add_sparam(
                f"\\delta z_{{\\mathrm{{lens{i}}}}}",
                f"lens{i}_delta_z",
                -0.05,
                0.05,
                0.001,
                0.0,
                0.0,
                Ncm.ParamType.FREE,
            )

        for i in range(4):
            model_builder.add_sparam(
                f"m_{{\\mathrm{{src{i}}}}}",
                f"src{i}_mult_bias",
                0.001,
                0.1,
                0.001,
                0.0,
                0.012,
                Ncm.ParamType.FREE,
            )

        FirecrownModel = register_model_class(model_builder)
        mset = config.get("model-set")
        assert isinstance(mset, Ncm.MSet)
        mset.set(FirecrownModel())

        numcosmo_yaml = output_path / f"numcosmo_{self.prefix}.yaml"
        builders_file = numcosmo_yaml.with_suffix(".builders.yaml")

        model_builders = Ncm.ObjDictStr.new()  # pylint: disable=no-value-for-parameter
        model_builders.add(model_name, model_builder)

        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        ser.dict_str_to_yaml_file(config, numcosmo_yaml.as_posix())
        ser.dict_str_to_yaml_file(model_builders, builders_file.as_posix())
