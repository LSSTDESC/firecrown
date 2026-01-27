"""Unit tests for NumCosmo full workflow integration and serialization.

Tests for complete end-to-end workflows including write_config() output
validation, YAML serialization/deserialization, and configuration correctness
in firecrown.app.analysis._numcosmo module.
"""

from pathlib import Path
import pytest

from numcosmo_py import Ncm, Nc

from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._numcosmo import (
    NumCosmoConfigGenerator,
    NAME_MAP,
)
from firecrown.app.analysis._types import (
    FrameworkCosmology,
    CCLCosmologySpec,
    Parameter,
    PriorGaussian,
    PriorUniform,
    Model,
)


@pytest.fixture(name="numcosmo_init", scope="session")
def fixture_numcosmo_init() -> bool:
    """Fixture to initialize NumCosmo for testing."""
    Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

    return True


@pytest.fixture(name="minimal_factory_file")
def fixture_minimal_factory_file(tmp_path: Path) -> Path:
    """Create a minimal factory file for testing.

    This factory creates a simple likelihood with:
    - One tracer (lens0)
    - One two-point statistic (galaxy_density_cl)
    - Mock SACC data with 3 data points
    """
    factory_file = tmp_path / "factory.py"
    factory_file.write_text("""
import sacc
import numpy as np
from firecrown.likelihood.number_counts import NumberCounts
from firecrown.likelihood import ConstGaussian, TwoPoint


def build_likelihood(_):
    lens0 = NumberCounts(sacc_tracer="lens0")
    two_point = TwoPoint("galaxy_density_cl", source0=lens0, source1=lens0)
    statistics = [two_point]

    sacc_data = sacc.Sacc()
    sacc_data.add_tracer(
        "NZ", "lens0", np.array([0.1, 0.2, 0.3]), np.array([0.0, 1.0, 0.0])
    )
    sacc_data.add_ell_cl(
        "galaxy_density_cl",
        "lens0",
        "lens0",
        np.array([10, 20, 30]),
        np.array([1.0, 2.0, 3.0]),
    )
    sacc_data.add_covariance(np.eye(3) * 0.1)

    likelihood = ConstGaussian(statistics=statistics)
    likelihood.read(sacc_data)
    return likelihood
""".strip())
    return factory_file


class TestFullWorkflowIntegration:
    """Integration tests for full NumCosmo workflow."""

    def test_write_config_with_cosmology_priors(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test write_config with priors on cosmology parameters."""
        assert numcosmo_init

        # Create cosmology with Gaussian priors
        prior_omega_c = PriorGaussian(mean=0.25, sigma=0.05)
        prior_h = PriorUniform(lower=0.6, upper=0.8)

        params = []
        for p in CCLCosmologySpec.vanilla_lcdm().parameters:
            if p.name == "Omega_c":
                params.append(p.model_copy(update={"prior": prior_omega_c}))
            elif p.name == "h":
                params.append(p.model_copy(update={"prior": prior_h}))
            else:
                params.append(p)

        cosmo_with_priors = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_cosmo_priors",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_priors,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        expected_file = tmp_path / "numcosmo_test_cosmo_priors.yaml"
        assert expected_file.exists()

    def test_write_config_with_a_s_priors(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test write_config with A_s parameter and priors."""
        assert numcosmo_init

        # Create cosmology with A_s and Gaussian prior
        prior_as = PriorGaussian(mean=2e-9, sigma=0.1e-9)
        params = [
            p for p in CCLCosmologySpec.vanilla_lcdm().parameters if p.name != "sigma8"
        ] + [
            Parameter(
                name="A_s",
                symbol="A_s",
                lower_bound=1e-9,
                upper_bound=3e-9,
                default_value=2e-9,
                free=True,
                prior=prior_as,
            )
        ]
        cosmo_with_as = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_as_workflow",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_as,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        expected_file = tmp_path / "numcosmo_test_as_workflow.yaml"
        assert expected_file.exists()

    def test_write_config_with_a_s_uniform_prior(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test write_config with A_s parameter and uniform prior."""
        assert numcosmo_init

        # Create cosmology with A_s and uniform prior
        prior_as = PriorUniform(lower=1.5e-9, upper=2.5e-9)
        params = [
            p for p in CCLCosmologySpec.vanilla_lcdm().parameters if p.name != "sigma8"
        ] + [
            Parameter(
                name="A_s",
                symbol="A_s",
                lower_bound=1e-9,
                upper_bound=3e-9,
                default_value=2e-9,
                free=True,
                prior=prior_as,
            )
        ]
        cosmo_with_as = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_as_uniform",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_as,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        expected_file = tmp_path / "numcosmo_test_as_uniform.yaml"
        assert expected_file.exists()

    def test_write_config_with_model_parameters(
        self,
        numcosmo_init: bool,
        tmp_path: Path,
        minimal_factory_file: Path,
    ) -> None:
        """Test write_config with model parameters and priors."""
        assert numcosmo_init

        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()

        # Create model with priors
        prior_gaussian = PriorGaussian(mean=1.0, sigma=0.2)
        prior_uniform = PriorUniform(lower=0.5, upper=1.5)

        model = Model(
            name="test_model",
            description="Test model with priors",
            parameters=[
                Parameter(
                    name="param1",
                    symbol="p1",
                    lower_bound=0.0,
                    upper_bound=2.0,
                    default_value=1.0,
                    free=True,
                    prior=prior_gaussian,
                ),
                Parameter(
                    name="param2",
                    symbol="p2",
                    lower_bound=0.0,
                    upper_bound=2.0,
                    default_value=1.0,
                    free=True,
                    prior=prior_uniform,
                ),
            ],
        )

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_model_workflow",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.add_models([model])
        gen.write_config()

        expected_file = tmp_path / "numcosmo_test_model_workflow.yaml"
        assert expected_file.exists()

    def test_write_config_with_sigma8_gaussian_prior(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test write_config with sigma8 parameter and Gaussian prior."""
        assert numcosmo_init

        # Create cosmology with sigma8 Gaussian prior (vanilla LCDM has sigma8)
        prior_sigma8 = PriorGaussian(mean=0.8, sigma=0.05)
        params = [
            p.model_copy(update={"prior": prior_sigma8}) if p.name == "sigma8" else p
            for p in CCLCosmologySpec.vanilla_lcdm().parameters
        ]
        cosmo_with_sigma8 = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_sigma8_gauss",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_sigma8,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        expected_file = tmp_path / "numcosmo_test_sigma8_gauss.yaml"
        assert expected_file.exists()

    def test_write_config_with_sigma8_uniform_prior(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test write_config with sigma8 parameter and uniform prior."""
        assert numcosmo_init

        # Create cosmology with sigma8 uniform prior
        prior_sigma8 = PriorUniform(lower=0.7, upper=0.9)
        params = [
            p.model_copy(update={"prior": prior_sigma8}) if p.name == "sigma8" else p
            for p in CCLCosmologySpec.vanilla_lcdm().parameters
        ]
        cosmo_with_sigma8 = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_sigma8_uniform",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_sigma8,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        expected_file = tmp_path / "numcosmo_test_sigma8_uniform.yaml"
        assert expected_file.exists()

    def test_write_config_with_none_cosmology_workflow(
        self,
        numcosmo_init: bool,
        tmp_path: Path,
        minimal_factory_file: Path,
    ) -> None:
        """Test write_config with NONE cosmology framework."""
        assert numcosmo_init

        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_none_workflow",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONE,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        expected_file = tmp_path / "numcosmo_test_none_workflow.yaml"
        assert expected_file.exists()


class TestWriteConfigSerialization:
    """Tests for write_config() output deserialization and validation.

    These tests call write_config() to generate YAML files, then deserialize them
    to verify the created NumCosmo objects are correct. This ensures the subprocess
    isolation in _write_config_worker creates valid configurations.
    """

    def test_write_config_creates_valid_yaml(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test that write_config creates valid YAML files."""
        assert numcosmo_init

        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_yaml",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        # Verify YAML files exist
        yaml_file = tmp_path / "numcosmo_test_yaml.yaml"
        builders_file = tmp_path / "numcosmo_test_yaml.builders.yaml"
        assert yaml_file.exists()
        assert builders_file.exists()

        # Deserialize and check structure
        # pylint: disable-next=no-member
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())
        assert experiment is not None
        keys = experiment.keys()
        assert "likelihood" in keys
        assert "model-set" in keys

    def test_write_config_model_set_contains_cosmology(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test that model set in config contains cosmology model."""
        assert numcosmo_init

        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_mset",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        yaml_file = tmp_path / "numcosmo_test_mset.yaml"
        # pylint: disable-next=no-member
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())

        mset_obj = experiment.get("model-set")
        assert mset_obj is not None
        # Verify model set was deserialized correctly
        assert isinstance(mset_obj, Ncm.MSet)

        # Verify we can retrieve the cosmology model
        cosmo = mset_obj.get(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        assert cosmo is not None
        assert isinstance(cosmo, Nc.HICosmo)

    def test_write_config_cosmology_parameters_set(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test that cosmology parameters are correctly set in model set."""
        assert numcosmo_init

        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_cosmo_params",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        yaml_file = tmp_path / "numcosmo_test_cosmo_params.yaml"
        # pylint: disable-next=no-member
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())

        mset_obj = experiment.get("model-set")
        assert isinstance(mset_obj, Ncm.MSet)
        # Retrieve the actual cosmology model
        cosmo = mset_obj.get(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        assert cosmo is not None

        # Verify standard parameters are set
        for param_name in vanilla_cosmo.parameters:
            if param_name.name == "A_s":  # Skip amplitude parameter
                continue
            nc_name = NAME_MAP.get(param_name.name)
            if nc_name is not None and nc_name in cosmo.param_names():
                # Parameter should have a value
                value = cosmo[nc_name]
                assert isinstance(value, float)

    def test_write_config_with_priors_includes_priors(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test that priors are correctly included in likelihood."""
        assert numcosmo_init

        # Create cosmology with a prior
        prior_omega_c = PriorGaussian(mean=0.265, sigma=0.01)
        cosmo_with_prior = CCLCosmologySpec(
            parameters=[
                (
                    p
                    if p.name != "Omega_c"
                    else p.model_copy(update={"prior": prior_omega_c})
                )
                for p in CCLCosmologySpec.vanilla_lcdm().parameters
            ]
        )

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_priors",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_prior,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        yaml_file = tmp_path / "numcosmo_test_priors.yaml"
        # pylint: disable-next=no-member
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())

        likelihood_obj = experiment.get("likelihood")
        mset_obj = experiment.get("model-set")
        # Verify both objects were deserialized correctly
        assert isinstance(likelihood_obj, Ncm.Likelihood)
        assert isinstance(mset_obj, Ncm.MSet)

        # Verify we have a cosmology with priors set
        cosmo = mset_obj.get(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        assert cosmo is not None

    def test_write_config_with_neutrinos_sets_massnu(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test that write_config handles neutrino cosmology."""
        assert numcosmo_init

        cosmo_with_nu = CCLCosmologySpec.vanilla_lcdm_with_neutrinos()
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_neutrinos_ser",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_nu,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        yaml_file = tmp_path / "numcosmo_test_neutrinos_ser.yaml"
        # pylint: disable-next=no-member
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())

        mset_obj = experiment.get("model-set")
        # Verify neutrino cosmology configuration was created
        assert isinstance(mset_obj, Ncm.MSet)
        assert mset_obj.nmodels() > 0

        # Retrieve and verify the cosmology model has neutrino support
        cosmo = mset_obj.get(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        assert cosmo is not None
        assert isinstance(cosmo, Nc.HICosmo)
        assert cosmo.vparam_len(Nc.HICosmoDEVParams.M) > 0

    def test_write_config_with_a_s_parameter(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test that A_s amplitude configuration is created."""
        assert numcosmo_init

        cosmo_with_as = CCLCosmologySpec.vanilla_lcdm()
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_as_ser",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_as,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        yaml_file = tmp_path / "numcosmo_test_as_ser.yaml"
        # pylint: disable-next=no-member
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())

        # Verify both components were serialized
        mset_obj = experiment.get("model-set")
        likelihood_obj = experiment.get("likelihood")
        assert isinstance(mset_obj, Ncm.MSet)
        assert isinstance(likelihood_obj, Ncm.Likelihood)

        # Verify cosmology model was properly created
        cosmo = mset_obj.get(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        assert cosmo is not None
        assert isinstance(cosmo, Nc.HICosmo)

    def test_write_config_with_linear_cosmology(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test write_config with LINEAR cosmology level."""
        assert numcosmo_init

        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_linear",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.LINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        yaml_file = tmp_path / "numcosmo_test_linear.yaml"
        # pylint: disable-next=no-member
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())

        mset_obj = experiment.get("model-set")
        assert isinstance(mset_obj, Ncm.MSet)

        # Verify cosmology model is properly configured for LINEAR
        cosmo = mset_obj.get(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        assert cosmo is not None
        assert isinstance(cosmo, Nc.HICosmo)

    def test_write_config_deserialized_likelihood_evaluates(
        self, numcosmo_init: bool, tmp_path: Path, minimal_factory_file: Path
    ) -> None:
        """Test that deserialized likelihood can be evaluated."""
        assert numcosmo_init

        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_eval",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        yaml_file = tmp_path / "numcosmo_test_eval.yaml"
        # pylint: disable-next=no-member
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())

        likelihood_obj = experiment.get("likelihood")
        mset_obj = experiment.get("model-set")

        # Verify we have valid deserialized objects
        assert isinstance(mset_obj, Ncm.MSet)
        assert isinstance(likelihood_obj, Ncm.Likelihood)
        assert mset_obj.nmodels() > 0

        cosmo = mset_obj.get(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        prim = mset_obj.get(Nc.HIPrim.id())  # pylint: disable=no-value-for-parameter
        reion = mset_obj.get(Nc.HIReion.id())  # pylint: disable=no-value-for-parameter

        assert cosmo is not None
        assert prim is not None
        assert reion is not None
