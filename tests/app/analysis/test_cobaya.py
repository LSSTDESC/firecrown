"""Unit tests for firecrown.app.analysis._cobaya module.

Tests Cobaya configuration generation, parameter mapping, and YAML output.
"""

from pathlib import Path
import yaml
import pytest

from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._cobaya import (
    create_config,
    _configure_parameter,
    _get_standard_params,
    add_models,
    write_config,
    CobayaConfigGenerator,
    NAME_MAP,
)
from firecrown.app.analysis._types import (
    Frameworks,
    FrameworkCosmology,
    CCLCosmologySpec,
    Parameter,
    PriorGaussian,
    PriorUniform,
    Model,
)


@pytest.fixture(name="vanilla_cosmo")
def fixture_vanilla_cosmo() -> CCLCosmologySpec:
    """Create vanilla LCDM cosmology spec for testing."""
    return CCLCosmologySpec.vanilla_lcdm()


@pytest.fixture(name="sample_parameter")
def fixture_sample_parameter() -> Parameter:
    """Create sample parameter for testing."""
    return Parameter(
        name="test_param",
        symbol=r"\theta",
        lower_bound=0.0,
        upper_bound=1.0,
        default_value=0.5,
        free=True,
        prior=None,
    )


@pytest.fixture(name="sample_model")
def fixture_sample_model(sample_parameter: Parameter) -> Model:
    """Create sample model for testing."""
    return Model(
        name="test_model",
        description="Test model",
        parameters=[sample_parameter],
    )


class TestNameMap:
    """Tests for parameter name mapping."""

    def test_name_map_completeness(self) -> None:
        """Test that NAME_MAP contains expected cosmological parameters."""
        expected_keys = {
            "Omega_c",
            "Omega_b",
            "Omega_k",
            "T_CMB",
            "h",
            "Neff",
            "m_nu",
            "w0",
            "wa",
            "sigma8",
            "A_s",
            "n_s",
        }
        assert set(NAME_MAP.keys()) == expected_keys

    def test_name_map_values(self) -> None:
        """Test that NAME_MAP values match expected Cobaya/CAMB names."""
        assert NAME_MAP["Omega_c"] == "omch2"
        assert NAME_MAP["Omega_b"] == "ombh2"
        assert NAME_MAP["h"] == "H0"
        assert NAME_MAP["sigma8"] == "sigma8"
        assert NAME_MAP["A_s"] == "As"
        assert NAME_MAP["n_s"] == "ns"
        assert NAME_MAP["w0"] == "w"
        assert NAME_MAP["wa"] == "wa"


class TestConfigureParameter:
    """Tests for _configure_parameter function."""

    def test_configure_fixed_parameter(self) -> None:
        """Test configuration of fixed (non-free) parameter."""
        param = Parameter(
            name="fixed_param",
            symbol=r"\theta_f",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=False,
            prior=None,
        )
        result = _configure_parameter(param)
        assert result == 0.5

    def test_configure_free_parameter_no_prior(self) -> None:
        """Test configuration of free parameter without explicit prior."""
        param = Parameter(
            name="free_param",
            symbol=r"\theta",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=True,
            prior=None,
        )
        result = _configure_parameter(param)
        assert isinstance(result, dict)
        assert result["ref"] == 0.5
        assert result["prior"]["min"] == 0.0
        assert result["prior"]["max"] == 1.0

    def test_configure_parameter_with_gaussian_prior(self) -> None:
        """Test configuration with Gaussian prior."""
        prior = PriorGaussian(mean=0.5, sigma=0.1)
        param = Parameter(
            name="gauss_param",
            symbol=r"\theta_g",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=True,
            prior=prior,
        )
        result = _configure_parameter(param)
        assert isinstance(result, dict)
        assert result["ref"] == 0.5
        assert result["prior"]["dist"] == "norm"
        assert result["prior"]["loc"] == 0.5
        assert result["prior"]["scale"] == 0.1

    def test_configure_parameter_with_uniform_prior(self) -> None:
        """Test configuration with uniform prior."""
        prior = PriorUniform(lower=0.2, upper=0.8)
        param = Parameter(
            name="unif_param",
            symbol=r"\theta_u",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=True,
            prior=prior,
        )
        result = _configure_parameter(param)
        assert isinstance(result, dict)
        assert result["ref"] == 0.5
        assert result["prior"]["min"] == 0.2
        assert result["prior"]["max"] == 0.8

    def test_configure_parameter_with_scale(self) -> None:
        """Test parameter scaling (e.g., h -> H0 * 100)."""
        param = Parameter(
            name="h",
            symbol=r"h",
            lower_bound=0.6,
            upper_bound=0.8,
            default_value=0.7,
            free=True,
            prior=None,
        )
        scale_map = {"h": 100.0}
        result = _configure_parameter(param, scale_map)
        assert isinstance(result, dict)
        assert result["ref"] == 70.0  # 0.7 * 100
        assert result["prior"]["min"] == 60.0  # 0.6 * 100
        assert result["prior"]["max"] == 80.0  # 0.8 * 100

    def test_configure_parameter_omega_c_scaling(self) -> None:
        """Test Omega_c scaling with h^2."""
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.2,
            upper_bound=0.3,
            default_value=0.25,
            free=True,
            prior=None,
        )
        h = 0.7
        scale_map = {"Omega_c": h**2}
        result = _configure_parameter(param, scale_map)
        assert isinstance(result, dict)
        assert result["ref"] == pytest.approx(0.25 * 0.49)
        assert result["prior"]["min"] == pytest.approx(0.2 * 0.49)
        assert result["prior"]["max"] == pytest.approx(0.3 * 0.49)

    def test_configure_parameter_gaussian_with_scale(self) -> None:
        """Test Gaussian prior with scaling."""
        prior = PriorGaussian(mean=0.7, sigma=0.05)
        param = Parameter(
            name="h",
            symbol=r"h",
            lower_bound=0.6,
            upper_bound=0.8,
            default_value=0.7,
            free=True,
            prior=prior,
        )
        scale_map = {"h": 100.0}
        result = _configure_parameter(param, scale_map)
        assert isinstance(result, dict)
        assert result["prior"]["dist"] == "norm"
        assert result["prior"]["loc"] == 70.0
        assert result["prior"]["scale"] == 5.0


class TestGetStandardParams:
    """Tests for _get_standard_params function."""

    def test_standard_params_none_cosmology(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test that NONE cosmology returns empty dict."""
        params = _get_standard_params(FrameworkCosmology.NONE, vanilla_cosmo)
        assert not params

    def test_standard_params_nonlinear_cosmology(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test parameter generation for nonlinear cosmology."""
        params = _get_standard_params(FrameworkCosmology.NONLINEAR, vanilla_cosmo)
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_standard_params_parameter_mapping(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test that CCL names are mapped to Cobaya names."""
        params = _get_standard_params(FrameworkCosmology.NONLINEAR, vanilla_cosmo)

        # Check that Cobaya names are used (not CCL names)
        assert "H0" in params  # mapped from "h"
        assert "omch2" in params  # mapped from "Omega_c"
        assert "ombh2" in params  # mapped from "Omega_b"

        # Check that CCL names are NOT in output
        assert "h" not in params
        assert "Omega_c" not in params
        assert "Omega_b" not in params

    def test_standard_params_h_scaling(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test that h is scaled by 100 to get H0."""
        params = _get_standard_params(FrameworkCosmology.NONLINEAR, vanilla_cosmo)
        h_param = vanilla_cosmo["h"]
        h0_value = params["H0"]

        if isinstance(h0_value, dict):
            assert h0_value["ref"] == pytest.approx(h_param.default_value * 100)
        else:
            assert h0_value == pytest.approx(h_param.default_value * 100)

    def test_standard_params_linear_cosmology(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test parameter generation for linear cosmology."""
        params = _get_standard_params(FrameworkCosmology.LINEAR, vanilla_cosmo)
        assert isinstance(params, dict)
        assert len(params) > 0


class TestAddModels:
    """Tests for add_models function."""

    def test_add_models_to_empty_config(self, sample_model: Model) -> None:
        """Test adding models to empty configuration."""
        config: dict = {"params": {}}
        add_models(config, [sample_model])
        assert "test_param" in config["params"]

    def test_add_models_preserves_existing_params(self, sample_model: Model) -> None:
        """Test that adding models preserves existing parameters."""
        config: dict = {"params": {"existing_param": 1.0}}
        add_models(config, [sample_model])
        assert "existing_param" in config["params"]
        assert "test_param" in config["params"]

    def test_add_multiple_models(self) -> None:
        """Test adding multiple models."""
        model1 = Model(
            name="model1",
            description="First model",
            parameters=[
                Parameter(
                    name="param1",
                    symbol="p1",
                    lower_bound=0.0,
                    upper_bound=1.0,
                    default_value=0.5,
                    free=True,
                )
            ],
        )
        model2 = Model(
            name="model2",
            description="Second model",
            parameters=[
                Parameter(
                    name="param2",
                    symbol="p2",
                    lower_bound=0.0,
                    upper_bound=2.0,
                    default_value=1.0,
                    free=True,
                )
            ],
        )
        config: dict = {"params": {}}
        add_models(config, [model1, model2])
        assert "param1" in config["params"]
        assert "param2" in config["params"]

    def test_add_models_empty_list(self) -> None:
        """Test adding empty model list."""
        config: dict = {"params": {"existing": 1.0}}
        add_models(config, [])
        assert config["params"] == {"existing": 1.0}


class TestWriteConfig:
    """Tests for write_config function."""

    def test_write_config_creates_file(self, tmp_path: Path) -> None:
        """Test that write_config creates YAML file."""
        config = {"test": "value", "nested": {"key": "val"}}
        output_file = tmp_path / "test.yaml"
        write_config(config, output_file)
        assert output_file.exists()

    def test_write_config_valid_yaml(self, tmp_path: Path) -> None:
        """Test that written file is valid YAML."""
        config = {"params": {"param1": 0.5}, "sampler": "mcmc"}
        output_file = tmp_path / "test.yaml"
        write_config(config, output_file)

        with output_file.open("r") as f:
            loaded = yaml.safe_load(f)

        assert loaded == config

    def test_write_config_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that write_config overwrites existing files."""
        output_file = tmp_path / "test.yaml"

        # Write first config
        config1 = {"version": 1}
        write_config(config1, output_file)

        # Write second config
        config2 = {"version": 2}
        write_config(config2, output_file)

        with output_file.open("r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["version"] == 2


class TestCreateConfig:
    """Tests for create_config function."""

    def test_create_config_structure(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test basic structure of created configuration."""
        config = create_config(
            factory_source=Path("/test/factory.py"),
            build_parameters=NamedParameters({}),
            likelihood_name="test_likelihood",
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "theory" in config
        assert "likelihood" in config
        assert "params" in config
        assert "sampler" in config

    def test_create_config_theory_section(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test theory section includes CAMB configuration."""
        config = create_config(
            factory_source=Path("/test/factory.py"),
            build_parameters=NamedParameters({}),
            likelihood_name="test_likelihood",
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "camb" in config["theory"]
        assert config["theory"]["camb"]["stop_at_error"] is True

    def test_create_config_likelihood_section(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test likelihood section structure."""
        factory_path = Path("/test/factory.py")
        build_params = NamedParameters({"sacc_file": "data.sacc"})

        config = create_config(
            factory_source=factory_path,
            build_parameters=build_params,
            likelihood_name="my_likelihood",
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "my_likelihood" in config["likelihood"]
        likelihood_cfg = config["likelihood"]["my_likelihood"]
        assert "external" in likelihood_cfg
        assert "firecrownIni" in likelihood_cfg
        assert "build_parameters" in likelihood_cfg
        assert "input_style" in likelihood_cfg
        assert likelihood_cfg["input_style"] == "CAMB"

    def test_create_config_sampler_section(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test sampler section defaults."""
        config = create_config(
            factory_source=Path("/test/factory.py"),
            build_parameters=NamedParameters({}),
            likelihood_name="test",
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert config["sampler"] == {"evaluate": None}

    def test_create_config_no_cosmology(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test configuration without cosmology (NONE mode)."""
        config = create_config(
            factory_source=Path("/test/factory.py"),
            build_parameters=NamedParameters({}),
            likelihood_name="test",
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONE,
        )

        # Should not have theory or params sections
        assert "theory" not in config
        assert config["params"] == {}

        # Should have likelihood
        assert "likelihood" in config
        assert "input_style" not in config["likelihood"]["test"]

    def test_create_config_absolute_path(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test absolute path handling in configuration."""
        factory_path = Path("/absolute/path/factory.py")
        config = create_config(
            factory_source=factory_path,
            build_parameters=NamedParameters({}),
            likelihood_name="test",
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        factory_ini = config["likelihood"]["test"]["firecrownIni"]
        assert factory_ini == str(factory_path.absolute())

    def test_create_config_relative_path(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test relative path handling in configuration."""
        factory_path = Path("factory.py")
        config = create_config(
            factory_source=factory_path,
            build_parameters=NamedParameters({}),
            likelihood_name="test",
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=False,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        factory_ini = config["likelihood"]["test"]["firecrownIni"]
        assert factory_ini == "factory.py"

    def test_create_config_build_parameters(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test that build parameters are included."""
        build_params = NamedParameters({"sacc_file": "data.sacc", "option": "value"})
        config = create_config(
            factory_source=Path("factory.py"),
            build_parameters=build_params,
            likelihood_name="test",
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        bp = config["likelihood"]["test"]["build_parameters"]
        assert bp["sacc_file"] == "data.sacc"
        assert bp["option"] == "value"


class TestCobayaConfigGenerator:
    """Tests for CobayaConfigGenerator class."""

    def test_generator_framework(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test that generator has correct framework."""
        assert vanilla_cosmo is not None
        assert CobayaConfigGenerator.framework == Frameworks.COBAYA

    def test_generator_initialization(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test generator initialization."""
        gen = CobayaConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert gen.framework == Frameworks.COBAYA
        assert gen.prefix == "test"
        assert gen.output_path == tmp_path

    def test_generator_write_config(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test that write_config creates YAML file."""
        gen = CobayaConfigGenerator(
            output_path=tmp_path,
            prefix="my_analysis",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        # Set required attributes
        gen.factory_source = tmp_path / "factory.py"
        gen.build_parameters = NamedParameters({"sacc_file": "data.sacc"})

        gen.write_config()

        expected_file = tmp_path / "cobaya_my_analysis.yaml"
        assert expected_file.exists()

    def test_generator_write_config_content(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test content of written configuration file."""
        gen = CobayaConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = tmp_path / "factory.py"
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        yaml_file = tmp_path / "cobaya_test.yaml"
        # Read as text and check structure (contains Python class references)
        content = yaml_file.read_text()

        assert "theory:" in content
        assert "likelihood:" in content
        assert "firecrown_test:" in content
        assert "camb:" in content
        assert "params:" in content

    def test_generator_write_config_with_models(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec, sample_model: Model
    ) -> None:
        """Test configuration includes model parameters."""
        gen = CobayaConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = tmp_path / "factory.py"
        gen.build_parameters = NamedParameters({})
        gen.models = [sample_model]
        gen.write_config()

        yaml_file = tmp_path / "cobaya_test.yaml"
        # Read as text to check model parameters (YAML contains Python class references)
        content = yaml_file.read_text()

        assert "test_param:" in content
