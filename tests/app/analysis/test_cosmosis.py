"""Unit tests for firecrown.app.analysis._cosmosis module.

Tests CosmoSIS configuration generation, INI file formatting, and parameter handling.
"""

from pathlib import Path
import configparser
import pytest

from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._cosmosis import (
    format_comment,
    add_comment_block,
    _add_cosmology_modules,
    _add_firecrown_likelihood,
    create_config,
    add_models,
    format_float,
    add_firecrown_model,
    create_values_config,
    add_model_priors,
    create_priors_config,
    CosmosisConfigGenerator,
    NAME_MAP,
    COSMOLOGICAL_PARAMETERS,
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
            "h",
            "w0",
            "wa",
            "sigma8",
            "A_s",
            "n_s",
            "Neff",
            "m_nu",
        }
        assert set(NAME_MAP.keys()) == expected_keys

    def test_name_map_values(self) -> None:
        """Test that NAME_MAP values match expected CosmoSIS names."""
        assert NAME_MAP["Omega_c"] == "omega_c"
        assert NAME_MAP["Omega_b"] == "omega_b"
        assert NAME_MAP["h"] == "h0"
        assert NAME_MAP["sigma8"] == "sigma_8"
        assert NAME_MAP["A_s"] == "A_s"
        assert NAME_MAP["n_s"] == "n_s"


class TestFormatComment:
    """Tests for format_comment function."""

    def test_format_comment_short_text(self) -> None:
        """Test formatting of short comment text."""
        result = format_comment("Short comment")
        assert result == [";; Short comment"]

    def test_format_comment_long_text(self) -> None:
        """Test wrapping of long comment text."""
        long_text = "This is a very long comment " * 10
        result = format_comment(long_text, width=88)

        assert len(result) > 1
        for line in result:
            assert line.startswith(";; ")
            assert len(line) <= 88

    def test_format_comment_custom_width(self) -> None:
        """Test comment formatting with custom width."""
        text = "A" * 100
        result = format_comment(text, width=50)

        for line in result:
            assert len(line) <= 50

    def test_format_comment_empty_text(self) -> None:
        """Test formatting empty comment."""
        # Any empty string should be formatted as an empty list
        result = format_comment("")
        assert result == []


class TestAddCommentBlock:
    """Tests for add_comment_block function."""

    def test_add_comment_block_new_section(self) -> None:
        """Test adding comment block to new section."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section("test_section")

        add_comment_block(cfg, "test_section", "Test comment")

        # Check that comment was added
        items = list(cfg["test_section"].keys())
        assert any(item.startswith(";; ") for item in items)

    def test_add_comment_block_multiline(self) -> None:
        """Test adding multiline comment block."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section("test_section")

        long_text = "This is a very long comment " * 10
        add_comment_block(cfg, "test_section", long_text)

        items = list(cfg["test_section"].keys())
        comment_lines = [item for item in items if item.startswith(";; ")]
        assert len(comment_lines) > 1


class TestFormatFloat:
    """Tests for format_float function."""

    def test_format_float_integer(self) -> None:
        """Test formatting integer values."""
        assert format_float(1.0) == "1.0"
        assert format_float(42.0) == "42.0"

    def test_format_float_decimal(self) -> None:
        """Test formatting decimal values."""
        result = format_float(0.67)
        assert "." in result
        assert result == "0.67"

    def test_format_float_scientific(self) -> None:
        """Test formatting values in scientific notation."""
        result = format_float(2.1e-9)
        assert "e" in result

    def test_format_float_small_value(self) -> None:
        """Test formatting very small values."""
        result = format_float(1e-10)
        assert "e" in result or "." in result

    def test_format_float_always_has_decimal_or_e(self) -> None:
        """Test that formatted floats always have . or e."""
        test_values = [0.0, 1.0, 10.0, 100.0, 0.5, 3.14, 1e-9]
        for value in test_values:
            result = format_float(value)
            assert "." in result or "e" in result


class TestAddCosmologyModules:
    """Tests for _add_cosmology_modules function."""

    def test_add_cosmology_modules_nonlinear(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test adding cosmology modules for nonlinear computation."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        _add_cosmology_modules(cfg, FrameworkCosmology.NONLINEAR, vanilla_cosmo)

        assert "consistency" in cfg
        assert "camb" in cfg
        assert cfg["camb"]["mode"] == "power"

    def test_add_cosmology_modules_background(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test adding cosmology modules for background computation."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        _add_cosmology_modules(cfg, FrameworkCosmology.BACKGROUND, vanilla_cosmo)

        assert "consistency" in cfg
        assert "camb" in cfg
        assert cfg["camb"]["mode"] == "background"

    def test_add_cosmology_modules_linear(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test adding cosmology modules for linear computation."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        _add_cosmology_modules(cfg, FrameworkCosmology.LINEAR, vanilla_cosmo)

        assert "camb" in cfg
        assert cfg["camb"]["mode"] == "power"

    def test_add_cosmology_modules_camb_parameters(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test that CAMB section includes required parameters."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        _add_cosmology_modules(cfg, FrameworkCosmology.NONLINEAR, vanilla_cosmo)

        camb_cfg = cfg["camb"]
        assert "file" in camb_cfg
        assert "mode" in camb_cfg
        assert "feedback" in camb_cfg


class TestAddFirecrownLikelihood:
    """Tests for _add_firecrown_likelihood function."""

    def test_add_firecrown_likelihood_basic(self) -> None:
        """Test adding Firecrown likelihood module."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        build_params = NamedParameters({})

        _add_firecrown_likelihood(cfg, "factory.py", build_params)

        assert "firecrown_likelihood" in cfg
        assert "file" in cfg["firecrown_likelihood"]
        assert "likelihood_source" in cfg["firecrown_likelihood"]

    def test_add_firecrown_likelihood_with_params(self) -> None:
        """Test adding likelihood with build parameters."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        build_params = NamedParameters({"sacc_file": "data.sacc", "option": "value"})

        _add_firecrown_likelihood(cfg, "factory.py", build_params)

        assert cfg["firecrown_likelihood"]["sacc_file"] == "data.sacc"
        assert cfg["firecrown_likelihood"]["option"] == "value"


class TestCreateConfig:
    """Tests for create_config function."""

    def test_create_config_structure(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test basic structure of created configuration."""
        values_ini = tmp_path / "values.ini"

        cfg = create_config(
            prefix="test",
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            values_path=values_ini,
            priors_path=None,
            output_path=tmp_path,
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "runtime" in cfg
        assert "pipeline" in cfg
        assert "output" in cfg
        assert "test" in cfg

    def test_create_config_with_cosmology(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test configuration includes cosmology modules."""
        values_ini = tmp_path / "values.ini"

        cfg = create_config(
            prefix="test",
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            values_path=values_ini,
            priors_path=None,
            output_path=tmp_path,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "consistency" in cfg
        assert "camb" in cfg
        assert "firecrown_likelihood" in cfg

    def test_create_config_no_cosmology(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test configuration without cosmology modules."""
        values_ini = tmp_path / "values.ini"

        cfg = create_config(
            prefix="test",
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            values_path=values_ini,
            priors_path=None,
            output_path=tmp_path,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONE,
        )

        assert "consistency" not in cfg
        assert "camb" not in cfg
        assert "firecrown_likelihood" in cfg

    def test_create_config_pipeline_modules(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test pipeline modules configuration."""
        values_ini = tmp_path / "values.ini"

        cfg = create_config(
            prefix="test",
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            values_path=values_ini,
            priors_path=None,
            output_path=tmp_path,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        modules = cfg["pipeline"]["modules"]
        assert "consistency" in modules
        assert "camb" in modules
        assert "firecrown_likelihood" in modules

    def test_create_config_with_priors(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test configuration with priors file."""
        values_ini = tmp_path / "values.ini"
        priors_ini = tmp_path / "priors.ini"

        cfg = create_config(
            prefix="test",
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            values_path=values_ini,
            priors_path=priors_ini,
            output_path=tmp_path,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "priors" in cfg["pipeline"]

    def test_create_config_absolute_path(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test configuration with absolute paths."""
        values_ini = tmp_path / "values.ini"

        cfg = create_config(
            prefix="test",
            factory_source=tmp_path / "factory.py",
            build_parameters=NamedParameters({}),
            values_path=values_ini,
            priors_path=None,
            output_path=tmp_path,
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        values_path = cfg["pipeline"]["values"]
        assert str(tmp_path) in values_path or values_path == str(values_ini.absolute())


class TestAddModels:
    """Tests for add_models function."""

    def test_add_models_single_model(self, sample_model: Model) -> None:
        """Test adding single model to configuration."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section("firecrown_likelihood")

        add_models(cfg, [sample_model])

        assert "sampling_parameters_sections" in cfg["firecrown_likelihood"]
        sections = cfg["firecrown_likelihood"]["sampling_parameters_sections"]
        assert "test_model" in sections

    def test_add_models_multiple_models(self) -> None:
        """Test adding multiple models."""
        model1 = Model(name="model1", description="First", parameters=[])
        model2 = Model(name="model2", description="Second", parameters=[])

        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section("firecrown_likelihood")

        add_models(cfg, [model1, model2])

        sections = cfg["firecrown_likelihood"]["sampling_parameters_sections"]
        assert "model1" in sections
        assert "model2" in sections

    def test_add_models_empty_list(self) -> None:
        """Test adding empty model list."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section("firecrown_likelihood")

        add_models(cfg, [])

        sections = cfg["firecrown_likelihood"]["sampling_parameters_sections"]
        assert sections == ""


class TestAddFirecrownModel:
    """Tests for add_firecrown_model function."""

    def test_add_firecrown_model_free_parameter(self, sample_model: Model) -> None:
        """Test adding model with free parameter."""
        cfg = configparser.ConfigParser(allow_no_value=True)

        add_firecrown_model(cfg, sample_model)

        assert "test_model" in cfg
        assert "test_param" in cfg["test_model"]
        # Free parameter should have "min start max" format
        value = cfg["test_model"]["test_param"]
        parts = value.split()
        assert len(parts) == 3

    def test_add_firecrown_model_fixed_parameter(self) -> None:
        """Test adding model with fixed parameter."""
        param = Parameter(
            name="fixed_param",
            symbol="f",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.7,
            free=False,
        )
        model = Model(name="fixed_model", description="Fixed", parameters=[param])
        cfg = configparser.ConfigParser(allow_no_value=True)

        add_firecrown_model(cfg, model)

        value = cfg["fixed_model"]["fixed_param"]
        # Fixed parameter should have single value
        assert "." in value
        assert " " not in value

    def test_add_firecrown_model_custom_section(self, sample_model: Model) -> None:
        """Test adding model with custom section name."""
        cfg = configparser.ConfigParser(allow_no_value=True)

        add_firecrown_model(cfg, sample_model, section="custom_section")

        assert "custom_section" in cfg
        assert "test_param" in cfg["custom_section"]

    def test_add_firecrown_model_with_name_map(self, sample_model: Model) -> None:
        """Test adding model with parameter name mapping."""
        cfg = configparser.ConfigParser(allow_no_value=True)
        name_map = {"test_param": "mapped_param"}

        add_firecrown_model(cfg, sample_model, name_map=name_map)

        assert "mapped_param" in cfg["test_model"]
        assert "test_param" not in cfg["test_model"]


class TestCreateValuesConfig:
    """Tests for create_values_config function."""

    def test_create_values_config_with_cosmology(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test creating values config with cosmology."""
        cfg = create_values_config(
            vanilla_cosmo, required_cosmology=FrameworkCosmology.NONLINEAR
        )

        assert COSMOLOGICAL_PARAMETERS in cfg
        # Check for mapped parameter names
        assert (
            "h0" in cfg[COSMOLOGICAL_PARAMETERS]
            or "omega_c" in cfg[COSMOLOGICAL_PARAMETERS]
        )

    def test_create_values_config_no_cosmology(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test creating values config without cosmology."""
        cfg = create_values_config(
            vanilla_cosmo, required_cosmology=FrameworkCosmology.NONE
        )

        assert COSMOLOGICAL_PARAMETERS not in cfg

    def test_create_values_config_with_models(
        self, vanilla_cosmo: CCLCosmologySpec, sample_model: Model
    ) -> None:
        """Test creating values config with models."""
        cfg = create_values_config(
            vanilla_cosmo,
            models=[sample_model],
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "test_model" in cfg
        assert "test_param" in cfg["test_model"]


class TestAddModelPriors:
    """Tests for add_model_priors function."""

    def test_add_model_priors_gaussian(self) -> None:
        """Test adding Gaussian priors."""
        prior = PriorGaussian(mean=0.5, sigma=0.1)
        param = Parameter(
            name="param_gauss",
            symbol="pg",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=True,
            prior=prior,
        )
        model = Model(name="gauss_model", description="Gaussian", parameters=[param])

        cfg = configparser.ConfigParser(allow_no_value=True)
        add_model_priors(cfg, model)

        assert "gauss_model" in cfg
        prior_str = cfg["gauss_model"]["param_gauss"]
        assert "gaussian" in prior_str

    def test_add_model_priors_uniform(self) -> None:
        """Test adding uniform priors."""
        prior = PriorUniform(lower=0.2, upper=0.8)
        param = Parameter(
            name="param_unif",
            symbol="pu",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=True,
            prior=prior,
        )
        model = Model(name="unif_model", description="Uniform", parameters=[param])

        cfg = configparser.ConfigParser(allow_no_value=True)
        add_model_priors(cfg, model)

        assert "unif_model" in cfg
        prior_str = cfg["unif_model"]["param_unif"]
        assert "uniform" in prior_str

    def test_add_model_priors_no_priors(self, sample_model: Model) -> None:
        """Test that models without priors are skipped."""
        cfg = configparser.ConfigParser(allow_no_value=True)

        add_model_priors(cfg, sample_model)

        # Should not add section if no priors
        assert "test_model" not in cfg


class TestCreatePriorsConfig:
    """Tests for create_priors_config function."""

    def test_create_priors_config_no_priors(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test that None is returned when no priors defined."""
        result = create_priors_config(
            vanilla_cosmo, required_cosmology=FrameworkCosmology.NONLINEAR
        )

        assert result is None

    def test_create_priors_config_with_priors(self) -> None:
        """Test creating priors config when priors are defined."""
        prior = PriorGaussian(mean=0.5, sigma=0.1)
        param = Parameter(
            name="Omega_c",
            symbol="Omega_c",
            lower_bound=0.2,
            upper_bound=0.3,
            default_value=0.25,
            free=True,
            prior=prior,
        )

        # Create a minimal cosmology spec with a prior
        cosmo = CCLCosmologySpec(
            parameters=[
                param,
                Parameter(
                    name="Omega_b",
                    symbol="Omega_b",
                    lower_bound=0.03,
                    upper_bound=0.07,
                    default_value=0.05,
                    free=True,
                ),
                Parameter(
                    name="h",
                    symbol="h",
                    lower_bound=0.6,
                    upper_bound=0.8,
                    default_value=0.67,
                    free=False,
                ),
                Parameter(
                    name="n_s",
                    symbol="n_s",
                    lower_bound=0.87,
                    upper_bound=1.07,
                    default_value=0.96,
                    free=False,
                ),
                Parameter(
                    name="sigma8",
                    symbol="sigma8",
                    lower_bound=0.6,
                    upper_bound=1.0,
                    default_value=0.81,
                    free=True,
                ),
                Parameter(
                    name="Omega_k",
                    symbol="Omega_k",
                    lower_bound=-0.2,
                    upper_bound=0.2,
                    default_value=0.0,
                    free=False,
                ),
                Parameter(
                    name="Neff",
                    symbol="Neff",
                    lower_bound=2.0,
                    upper_bound=5.0,
                    default_value=3.046,
                    free=False,
                ),
                Parameter(
                    name="m_nu",
                    symbol="m_nu",
                    lower_bound=0.0,
                    upper_bound=5.0,
                    default_value=0.0,
                    free=False,
                ),
                Parameter(
                    name="w0",
                    symbol="w0",
                    lower_bound=-3.0,
                    upper_bound=0.0,
                    default_value=-1.0,
                    free=False,
                ),
                Parameter(
                    name="wa",
                    symbol="wa",
                    lower_bound=-1.0,
                    upper_bound=1.0,
                    default_value=0.0,
                    free=False,
                ),
            ]
        )

        result = create_priors_config(
            cosmo, required_cosmology=FrameworkCosmology.NONLINEAR
        )

        assert result is not None
        assert COSMOLOGICAL_PARAMETERS in result

    def test_create_priors_config_with_framework_none(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test priors config when required_cosmology is NONE."""
        # Add a prior to one parameter in vanilla cosmology
        prior = PriorGaussian(mean=0.25, sigma=0.05)
        param_with_prior = Parameter(
            name="Omega_c",
            symbol="Omega_c",
            lower_bound=0.2,
            upper_bound=0.3,
            default_value=0.25,
            free=True,
            prior=prior,
        )

        # Create new spec with the prior
        params = [
            param_with_prior if p.name == "Omega_c" else p
            for p in vanilla_cosmo.parameters
        ]
        cosmo = CCLCosmologySpec(parameters=params)

        # When required_cosmology is NONE, cosmo should be added to models list
        result = create_priors_config(cosmo, required_cosmology=FrameworkCosmology.NONE)

        assert result is not None
        # The prior should be in some section
        found = False
        for section in result.sections():
            if "Omega_c" in result[section]:
                found = True
                break
        assert found

    def test_create_priors_config_with_models(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test priors config with additional models."""
        # Create a model with priors
        prior = PriorGaussian(mean=1.0, sigma=0.1)
        param_with_prior = Parameter(
            name="test_param",
            symbol="test_param",
            lower_bound=0.5,
            upper_bound=1.5,
            default_value=1.0,
            free=True,
            prior=prior,
        )
        model = Model(
            name="test_model", description="Test model", parameters=[param_with_prior]
        )

        result = create_priors_config(
            vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            models=[model],
        )

        assert result is not None
        # The model section should exist
        assert "test_model" in result.sections()


class TestCosmosisConfigGenerator:
    """Tests for CosmosisConfigGenerator class."""

    def test_generator_framework(self) -> None:
        """Test that generator has correct framework."""
        assert CosmosisConfigGenerator.framework == Frameworks.COSMOSIS

    def test_generator_initialization(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test generator initialization."""
        gen = CosmosisConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert gen.framework == Frameworks.COSMOSIS
        assert gen.prefix == "test"

    def test_generator_write_config(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test that write_config creates INI files."""
        gen = CosmosisConfigGenerator(
            output_path=tmp_path,
            prefix="my_analysis",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = tmp_path / "factory.py"
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        # Check that files were created
        assert (tmp_path / "cosmosis_my_analysis.ini").exists()
        assert (tmp_path / "cosmosis_my_analysis_values.ini").exists()

    def test_generator_write_config_with_priors(self, tmp_path: Path) -> None:
        """Test configuration with priors file."""
        # Create cosmology with priors
        prior = PriorGaussian(mean=0.25, sigma=0.05)
        param = Parameter(
            name="Omega_c",
            symbol="Omega_c",
            lower_bound=0.2,
            upper_bound=0.3,
            default_value=0.25,
            free=True,
            prior=prior,
        )

        cosmo = CCLCosmologySpec(
            parameters=[
                param,
                Parameter(
                    name="Omega_b",
                    symbol="Omega_b",
                    lower_bound=0.03,
                    upper_bound=0.07,
                    default_value=0.05,
                    free=True,
                ),
                Parameter(
                    name="h",
                    symbol="h",
                    lower_bound=0.6,
                    upper_bound=0.8,
                    default_value=0.67,
                    free=False,
                ),
                Parameter(
                    name="n_s",
                    symbol="n_s",
                    lower_bound=0.87,
                    upper_bound=1.07,
                    default_value=0.96,
                    free=False,
                ),
                Parameter(
                    name="sigma8",
                    symbol="sigma8",
                    lower_bound=0.6,
                    upper_bound=1.0,
                    default_value=0.81,
                    free=True,
                ),
                Parameter(
                    name="Omega_k",
                    symbol="Omega_k",
                    lower_bound=-0.2,
                    upper_bound=0.2,
                    default_value=0.0,
                    free=False,
                ),
                Parameter(
                    name="Neff",
                    symbol="Neff",
                    lower_bound=2.0,
                    upper_bound=5.0,
                    default_value=3.046,
                    free=False,
                ),
                Parameter(
                    name="m_nu",
                    symbol="m_nu",
                    lower_bound=0.0,
                    upper_bound=5.0,
                    default_value=0.0,
                    free=False,
                ),
                Parameter(
                    name="w0",
                    symbol="w0",
                    lower_bound=-3.0,
                    upper_bound=0.0,
                    default_value=-1.0,
                    free=False,
                ),
                Parameter(
                    name="wa",
                    symbol="wa",
                    lower_bound=-1.0,
                    upper_bound=1.0,
                    default_value=0.0,
                    free=False,
                ),
            ]
        )

        gen = CosmosisConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = tmp_path / "factory.py"
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        # Priors file should be created
        assert (tmp_path / "cosmosis_test_priors.ini").exists()

    def test_generator_write_config_with_models(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec, sample_model: Model
    ) -> None:
        """Test configuration includes model parameters."""
        gen = CosmosisConfigGenerator(
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

        # Check values file contains model section
        values_ini = tmp_path / "cosmosis_test_values.ini"
        cfg = configparser.ConfigParser()
        cfg.read(values_ini)

        assert "test_model" in cfg
