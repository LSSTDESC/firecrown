"""DES Y1 3x2pt analysis example generator.

Generates a complete DES Y1 3x2pt analysis example with cosmic shear,
galaxy-galaxy lensing, and galaxy clustering.
"""

from typing import ClassVar, Annotated, assert_never
from types import ModuleType
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import typer

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.ccl_factory import PoweSpecAmplitudeParameter
from ..analysis import (
    AnalysisBuilder,
    Model,
    Parameter,
    FrameworkCosmology,
    download_from_url,
    copy_template,
)
from . import (
    _des_y1_3x2pt_template,
    _des_y1_3x2pt_pt_template,
    _des_y1_cosmic_shear_tatt_template,
    _des_y1_cosmic_shear_hmia_template,
    _des_y1_cosmic_shear_pk_modifier_template,
)


class DESY1FactoryType(str, Enum):
    """Available factory implementations for DES Y1 3x2pt analysis."""

    STANDARD = "standard"
    PT = "pt"
    TATT = "tatt"
    HMIA = "hmia"
    PK_MODIFIER = "pk_modifier"
    YAML_DEFAULT = "yaml_default"
    YAML_PURE_CCL = "yaml_pure_ccl"
    YAML_MU_SIGMA = "yaml_mu_sigma"


FACTORY_TYPE_HELP = """\
Factory implementation to use for generating the DES Y1 3x2pt analysis.
The factory determines how the analysis pipeline will be constructed.

Available options:\n
  - standard:       uses the default implementation template\n
  - pt:             uses perturbation theory calculations template\n
  - tatt:           uses TATT intrinsic alignment model\n
  - hmia:           uses halo model intrinsic alignment\n
  - pk_modifier:    uses power spectrum modifier for baryonic effects\n
  - yaml_default:   uses YAML-based factory configuration (default)\n
  - yaml_pure_ccl:  uses YAML-based factory with pure CCL mode\n
  - yaml_mu_sigma:  uses YAML-based factory with mu-sigma ISiTGR mode
"""


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
                "Prefix for generated filenames (e.g., 'des_y1_3x2pt' creates "
                "'des_y1_3x2pt.sacc')"
            ),
            show_default=True,
        ),
    ] = "des_y1_3x2pt"

    factory_type: Annotated[
        DESY1FactoryType,
        typer.Option(
            help=FACTORY_TYPE_HELP,
            show_default=True,
            case_sensitive=False,
            rich_help_panel="Factory Options",
            metavar="FACTORY_TYPE",
        ),
    ] = DESY1FactoryType.STANDARD

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

    def generate_factory(self, output_path: Path, sacc: Path) -> str | Path:
        """Copy DES Y1 3x2pt factory template or generate YAML config.

        :param output_path: Output directory
        :param _sacc: SACC file path (unused)
        :return: Path to factory file or YAML config
        """
        # YAML-based factories
        if self.factory_type in (
            DESY1FactoryType.YAML_DEFAULT,
            DESY1FactoryType.YAML_PURE_CCL,
            DESY1FactoryType.YAML_MU_SIGMA,
        ):
            output_file = output_path / f"{self.prefix}_experiment.yaml"
            yaml_content = self._get_yaml_config(sacc)
            output_file.write_text(yaml_content)
            return "firecrown.likelihood.factories.build_two_point_likelihood"

        # Python template-based factories
        template: ModuleType
        output_file = output_path / f"{self.prefix}_factory.py"
        match self.factory_type:
            case DESY1FactoryType.PT:
                template = _des_y1_3x2pt_pt_template
            case DESY1FactoryType.TATT:
                template = _des_y1_cosmic_shear_tatt_template
            case DESY1FactoryType.HMIA:
                template = _des_y1_cosmic_shear_hmia_template
            case DESY1FactoryType.PK_MODIFIER:
                template = _des_y1_cosmic_shear_pk_modifier_template
            case DESY1FactoryType.STANDARD:
                template = _des_y1_3x2pt_template
            case _:
                raise ValueError(f"Unknown factory type: {self.factory_type}")

        copy_template(template, output_file)
        return output_file

    def _get_yaml_config(self, sacc: Path) -> str:
        """Generate YAML configuration content based on factory type.

        :return: YAML configuration as string
        """
        sacc_path_str = (
            sacc.absolute().as_posix() if self.use_absolute_path else sacc.name
        )
        base_config = f"""---

data_source:
  sacc_data_file: {sacc_path_str}

two_point_factory:
  correlation_space: real
  number_counts_factories:
    - type_source: default
      global_systematics: []
      per_bin_systematics:
        - type: PhotoZShiftFactory
  weak_lensing_factories:
    - type_source: default
      global_systematics:
        - alphag: 1
          type: LinearAlignmentSystematicFactory
      per_bin_systematics:
        - type: MultiplicativeShearBiasFactory
        - type: PhotoZShiftFactory
"""

        if self.factory_type == DESY1FactoryType.YAML_PURE_CCL:
            return (
                base_config
                + """
ccl_factory:
  creation_mode: 'pure_ccl_mode'
  require_nonlinear_pk: true
"""
            )
        if self.factory_type == DESY1FactoryType.YAML_MU_SIGMA:
            return (
                base_config
                + """
ccl_factory:
  creation_mode: 'mu_sigma_isitgr'
  require_nonlinear_pk: true
"""
            )

        return (
            base_config
            + """
ccl_factory:
  require_nonlinear_pk: true
"""
        )

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Return build parameters for likelihood construction.

        :param sacc_path: Path to the SACC data file
        :return: Named parameters with sacc_file and optional config path
        """
        params = {"sacc_file": self.get_sacc_file(sacc_path)}

        # For YAML-based factories, add likelihood_config parameter
        if self.factory_type in (
            DESY1FactoryType.YAML_DEFAULT,
            DESY1FactoryType.YAML_PURE_CCL,
            DESY1FactoryType.YAML_MU_SIGMA,
        ):
            experiment_path = sacc_path.parent / f"{self.prefix}_experiment.yaml"
            params["likelihood_config"] = (
                experiment_path.absolute().as_posix()
                if self.use_absolute_path
                else experiment_path.name
            )

        return NamedParameters(params)

    def _get_ccl_cosmo_params(self) -> Model:
        return Model(
            name="firecrown_ccl_cosmo",
            description="CCL cosmology parameters",
            parameters=[
                Parameter("Omega_c", r"\Omega_{c}", 0.20, 0.35, 0.05, 0.0, 0.25, True),
                Parameter("Omega_b", r"\Omega_{b}", 0.03, 0.06, 0.01, 0.0, 0.05, True),
                Parameter("h", r"h", 0.60, 0.80, 0.05, 0.0, 0.73, False),
                Parameter("n_s", r"n_{s}", 0.90, 1.10, 0.05, 0.0, 0.96, False),
                Parameter("Omega_k", r"\Omega_{k}", -0.2, 0.2, 0.05, 0.0, 0.0, False),
                Parameter("w0", r"w_{0}", -1.5, -0.5, 0.05, 0.0, -1.0, False),
                Parameter("wa", r"w_{a}", -1.0, 1.0, 0.05, 0.0, 0.0, False),
                Parameter("T_CMB", r"T_{CMB}", 2.7, 2.8, 0.05, 0.0, 2.725, False),
                Parameter("Neff", r"N_{eff}", 3.0, 4.0, 0.05, 0.0, 3.046, False),
                Parameter("m_nu", r"m_{\nu}", 0.0, 0.1, 0.05, 0.0, 0.0, False),
                Parameter("sigma8", r"\sigma_{8}", 0.7, 0.9, 0.05, 0.0, 0.8, False),
            ],
        )

    def get_models(self) -> list[Model]:
        """Define DES Y1 systematic and bias parameters.

        :return: Model with IA, photo-z, bias, and multiplicative bias parameters
        """
        params_tatt: list[tuple[str, str, float, float, float, float, float, bool]] = [
            ("ia_a_1", r"A_{\mathrm{IA},1}", 0.9, 1.2, 0.05, 0.0, 1.0, True),
            ("ia_a_2", r"A_{\mathrm{IA},2}", 0.4, 0.6, 0.05, 0.0, 0.5, True),
            ("ia_a_d", r"A_{\mathrm{IA},d}", 0.4, 0.6, 0.05, 0.0, 0.5, True),
            ("ia_zpiv_1", r"z_{\mathrm{piv},1}", 0.0, 1.0, 0.05, 0.0, 0.62, False),
            ("ia_zpiv_2", r"z_{\mathrm{piv},2}", 0.0, 1.0, 0.05, 0.0, 0.62, False),
            ("ia_zpiv_d", r"z_{\mathrm{piv},d}", 0.0, 1.0, 0.05, 0.0, 0.62, False),
            ("ia_alphaz_1", r"\alpha_{z,1}", -5.0, 5.0, 0.05, 0.0, 0.0, False),
            ("ia_alphaz_2", r"\alpha_{z,2}", -5.0, 5.0, 0.05, 0.0, 0.0, False),
            ("ia_alphaz_d", r"\alpha_{z,d}", -5.0, 5.0, 0.05, 0.0, 0.0, False),
            ("src0_delta_z", r"\delta z_0", -0.05, 0.05, 0.005, 0.0, 0.0, True),
        ]
        params_hmia: list[tuple[str, str, float, float, float, float, float, bool]] = [
            ("ia_a_1h", r"A_{\mathrm{IA},1h}", 0.0, 0.01, 0.001, 0.0, 0.001, True),
            ("ia_a_2h", r"A_{\mathrm{IA},2h}", 0.0, 2.0, 0.1, 0.0, 1.0, True),
            ("src0_delta_z", r"\delta z_0", -0.05, 0.05, 0.005, 0.0, 0.0, True),
        ]
        params_pk_mod: list[
            tuple[str, str, float, float, float, float, float, bool]
        ] = [
            ("f_bar", r"f_{\mathrm{bar}}", 0.0, 1.0, 0.05, 0.0, 0.5, True),
            ("src0_delta_z", r"\delta z_0", -0.05, 0.05, 0.005, 0.0, 0.0, True),
        ]
        params_yaml_mu_sigma: list[
            tuple[str, str, float, float, float, float, float, bool]
        ] = [
            ("mg_musigma_mu", r"\mu_{\mathrm{MG}}", 0.8, 1.2, 0.05, 0.0, 1.0, True),
            (
                "mg_musigma_sigma",
                r"\sigma_{\mathrm{MG}}",
                0.8,
                1.2,
                0.05,
                0.0,
                1.0,
                True,
            ),
            ("mg_musigma_c1", r"c_1", 0.8, 1.2, 0.05, 0.0, 1.0, False),
            ("mg_musigma_c2", r"c_2", 0.8, 1.2, 0.05, 0.0, 1.0, False),
            ("mg_musigma_lambda0", r"\lambda_0", 0.8, 1.2, 0.05, 0.0, 1.0, False),
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
        params_std: list[tuple[str, str, float, float, float, float, float, bool]] = [
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
        params_pt: list[tuple[str, str, float, float, float, float, float, bool]] = [
            ("ia_a_1", r"A_{\mathrm{IA},1}", 0.9, 1.2, 0.05, 0.0, 1.0, True),
            ("ia_a_2", r"A_{\mathrm{IA},2}", 0.4, 0.6, 0.05, 0.0, 0.5, True),
            ("ia_a_d", r"A_{\mathrm{IA},d}", 0.4, 0.6, 0.05, 0.0, 0.5, True),
            ("ia_zpiv_1", r"z_{\mathrm{piv},1}", 0.0, 1.0, 0.05, 0.0, 0.62, False),
            ("ia_zpiv_2", r"z_{\mathrm{piv},2}", 0.0, 1.0, 0.05, 0.0, 0.62, False),
            ("ia_zpiv_d", r"z_{\mathrm{piv},d}", 0.0, 1.0, 0.05, 0.0, 0.62, False),
            ("ia_alphaz_1", r"\alpha_{z,1}", -5.0, 5.0, 0.05, 0.0, 0.0, False),
            ("ia_alphaz_2", r"\alpha_{z,2}", -5.0, 5.0, 0.05, 0.0, 0.0, False),
            ("ia_alphaz_d", r"\alpha_{z,d}", -5.0, 5.0, 0.05, 0.0, 0.0, False),
            ("lens0_b_2", r"b_2^0", 0.8, 1.2, 0.05, 0.0, 1.0, True),
            ("lens0_b_s", r"b_s^0", 0.8, 1.2, 0.05, 0.0, 1.0, True),
            ("lens0_mag_bias", r"s_{\mathrm{mag}}^0", 0.8, 1.2, 0.05, 0.0, 1.0, True),
            ("lens0_bias", r"b_0", 0.8, 3.0, 0.05, 0.0, 2.0, True),
            ("src0_delta_z", r"\delta z_{src0}", -0.05, 0.05, 0.005, 0.0, -0.001, True),
            ("lens0_delta_z", r"\delta z_{lens0}", -0.05, 0.05, 0.05, 0.0, 0.001, True),
        ]

        match self.factory_type:
            case DESY1FactoryType.TATT:
                parameters = params_tatt
            case DESY1FactoryType.HMIA:
                parameters = params_hmia
            case DESY1FactoryType.PK_MODIFIER:
                parameters = params_pk_mod
            case DESY1FactoryType.PT:
                parameters = params_pt
            case DESY1FactoryType.YAML_MU_SIGMA:
                parameters = params_yaml_mu_sigma
            case DESY1FactoryType.YAML_DEFAULT | DESY1FactoryType.YAML_PURE_CCL:
                parameters = params_std
            case DESY1FactoryType.STANDARD:
                parameters = params_std
            case _ as unreachable:
                assert_never(unreachable)

        models: list[Model] = []
        if self.factory_type in (
            DESY1FactoryType.YAML_PURE_CCL,
            DESY1FactoryType.YAML_MU_SIGMA,
        ):
            # Pure CCL and mu-sigma factories required CCL cosmological parameters
            models.append(self._get_ccl_cosmo_params())

        firecrown_model = Model(
            name=f"firecrown_{self.prefix}_{self.factory_type.value}",
            description="DES Y1 3x2pt parameters",
            parameters=[Parameter(*param) for param in parameters],
        )

        models.append(firecrown_model)
        return models

    def required_cosmology(self):
        """Cosmology requirement level."""
        if self.factory_type in (
            DESY1FactoryType.YAML_PURE_CCL,
            DESY1FactoryType.YAML_MU_SIGMA,
        ):
            return FrameworkCosmology.NONE
        return FrameworkCosmology.NONLINEAR

    def amplitude_parameter(self):
        """Return power spectrum amplitude parameter."""
        return PoweSpecAmplitudeParameter.SIGMA8
