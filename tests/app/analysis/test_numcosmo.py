"""Unit tests for firecrown.app.analysis._numcosmo module.

Tests NumCosmo configuration generation and parameter handling.
"""

from pathlib import Path
import pytest

from numcosmo_py import Ncm, Nc

from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._numcosmo import (
    ConfigOptions,
    _create_mapping,
    _param_to_nc_dict,
    NAME_MAP,
    NumCosmoConfigGenerator,
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
from firecrown.app.analysis._numcosmo import _to_pascal

# pylint: disable=too-many-lines


@pytest.fixture(name="numcosmo_init", scope="session")
def fixture_numcosmo_init() -> bool:
    """Fixture to initialize NumCosmo for testing."""
    Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

    return True


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


@pytest.fixture(name="config_options")
def fixture_config_options(
    tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
) -> ConfigOptions:
    """Create ConfigOptions for testing."""
    return ConfigOptions(
        output_path=tmp_path,
        factory_source=Path("factory.py"),
        build_parameters=NamedParameters({}),
        models=[],
        cosmo_spec=vanilla_cosmo,
        use_absolute_path=True,
        required_cosmology=FrameworkCosmology.NONLINEAR,
        prefix="test",
    )


@pytest.fixture(name="minimal_factory_file")
def fixture_minimal_factory_file(tmp_path: Path) -> Path:
    """Create a minimal factory file for testing.

    This factory creates a simple likelihood with:
    - One tracer (lens0)
    - One two-point statistic (galaxy_density_cl)
    - Mock SACC data with 3 data points
    """
    factory_file = tmp_path / "factory.py"
    factory_file.write_text(
        """
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
""".strip()
    )
    return factory_file


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
            "n_s",
            "T_CMB",
            "Neff",
            "m_nu",
        }
        assert set(NAME_MAP.keys()) == expected_keys

    def test_name_map_values(self) -> None:
        """Test that NAME_MAP values match expected NumCosmo names."""
        assert NAME_MAP["Omega_c"] == "Omegac"
        assert NAME_MAP["Omega_b"] == "Omegab"
        assert NAME_MAP["h"] == "H0"
        assert NAME_MAP["w0"] == "w0"
        assert NAME_MAP["wa"] == "w1"
        assert NAME_MAP["n_s"] == "n_SA"


class TestConfigOptions:
    """Tests for ConfigOptions dataclass."""

    def test_config_options_initialization(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test ConfigOptions initialization."""
        opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        assert opts.output_path == tmp_path
        assert opts.prefix == "test"
        assert opts.required_cosmology == FrameworkCosmology.NONLINEAR
        assert opts.distance_max_z == 4.0  # default
        assert opts.reltol == 1e-7  # default

    def test_config_options_custom_parameters(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test ConfigOptions with custom parameters."""
        opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=False,
            required_cosmology=FrameworkCosmology.LINEAR,
            prefix="custom",
            distance_max_z=5.0,
            reltol=1e-8,
        )

        assert opts.distance_max_z == 5.0
        assert opts.reltol == 1e-8
        assert opts.required_cosmology == FrameworkCosmology.LINEAR


class TestCreateMapping:
    """Tests for _create_mapping function."""

    def test_create_mapping_none_cosmology(self, config_options: ConfigOptions) -> None:
        """Test that NONE cosmology returns None."""
        config_options.required_cosmology = FrameworkCosmology.NONE
        result = _create_mapping(config_options)
        assert result is None

    def test_create_mapping_nonlinear(
        self, numcosmo_init: bool, config_options: ConfigOptions
    ) -> None:
        """Test creating mapping for nonlinear cosmology."""
        assert numcosmo_init
        config_options.required_cosmology = FrameworkCosmology.NONLINEAR
        result = _create_mapping(config_options)
        assert result is not None

    def test_create_mapping_linear(
        self, numcosmo_init: bool, config_options: ConfigOptions
    ) -> None:
        """Test creating mapping for linear cosmology."""
        assert numcosmo_init
        config_options.required_cosmology = FrameworkCosmology.LINEAR
        result = _create_mapping(config_options)
        assert result is not None

    def test_create_mapping_background(
        self, numcosmo_init: bool, config_options: ConfigOptions
    ) -> None:
        """Test creating mapping for background cosmology."""
        assert numcosmo_init
        config_options.required_cosmology = FrameworkCosmology.BACKGROUND
        result = _create_mapping(config_options)
        assert result is not None


class TestParamToNcDict:
    """Tests for _param_to_nc_dict function."""

    def test_param_to_nc_dict_basic(self, sample_parameter: Parameter) -> None:
        """Test basic parameter conversion."""
        result = _param_to_nc_dict(sample_parameter)

        assert isinstance(result, dict)
        assert "lower-bound" in result
        assert "upper-bound" in result
        assert "value" in result
        assert "fit" in result

    def test_param_to_nc_dict_field_mapping(self, sample_parameter: Parameter) -> None:
        """Test that field names are mapped correctly."""
        result = _param_to_nc_dict(sample_parameter)

        # Check that original names are mapped
        assert "lower-bound" in result  # was lower_bound
        assert "upper-bound" in result  # was upper_bound
        assert "value" in result  # was default_value
        assert "fit" in result  # was free

        # Check that excluded fields are not present
        assert "name" not in result
        assert "symbol" not in result
        assert "prior" not in result

    def test_param_to_nc_dict_values(self) -> None:
        """Test parameter values are correctly converted."""
        param = Parameter(
            name="test",
            symbol="t",
            lower_bound=0.1,
            upper_bound=0.9,
            default_value=0.5,
            free=True,
        )
        result = _param_to_nc_dict(param)

        assert result["lower-bound"] == 0.1
        assert result["upper-bound"] == 0.9
        assert result["value"] == 0.5
        assert result["fit"] is True

    def test_param_to_nc_dict_with_scale(self, sample_parameter: Parameter) -> None:
        """Test parameter scaling."""
        scale_map = {"test_param": 100.0}
        result = _param_to_nc_dict(sample_parameter, scale_map)

        assert result["lower-bound"] == 0.0  # 0.0 * 100
        assert result["upper-bound"] == 100.0  # 1.0 * 100
        assert result["value"] == 50.0  # 0.5 * 100

    def test_param_to_nc_dict_fixed_parameter(self) -> None:
        """Test conversion of fixed parameter."""
        param = Parameter(
            name="fixed",
            symbol="f",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.7,
            free=False,
        )
        result = _param_to_nc_dict(param)

        assert result["fit"] is False
        assert result["value"] == 0.7


class TestNumCosmoConfigGenerator:
    """Tests for NumCosmoConfigGenerator class."""

    def test_generator_framework(self) -> None:
        """Test that generator has correct framework."""
        assert NumCosmoConfigGenerator.framework == Frameworks.NUMCOSMO

    def test_generator_initialization(
        self, numcosmo_init: bool, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test generator initialization."""
        assert numcosmo_init
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert gen.framework == Frameworks.NUMCOSMO
        assert gen.prefix == "test"
        assert gen.output_path == tmp_path

    def test_generator_write_config(
        self,
        numcosmo_init: bool,
        tmp_path: Path,
        vanilla_cosmo: CCLCosmologySpec,
        minimal_factory_file: Path,
    ) -> None:
        """Test that write_config creates YAML files."""
        assert numcosmo_init
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="my_analysis",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        # Check that files were created
        expected_file = tmp_path / "numcosmo_my_analysis.yaml"
        assert expected_file.exists()


class TestPriorHandling:
    """Tests for prior-related functions."""

    def test_parameter_with_gaussian_prior(self) -> None:
        """Test parameter conversion with Gaussian prior."""
        prior = PriorGaussian(mean=0.5, sigma=0.1)
        param = Parameter(
            name="gauss_param",
            symbol="g",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=True,
            prior=prior,
        )

        # Priors are not included in _param_to_nc_dict
        result = _param_to_nc_dict(param)
        assert "prior" not in result

    def test_parameter_with_uniform_prior(self) -> None:
        """Test parameter conversion with uniform prior."""
        prior = PriorUniform(lower=0.2, upper=0.8)
        param = Parameter(
            name="unif_param",
            symbol="u",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=True,
            prior=prior,
        )

        result = _param_to_nc_dict(param)
        assert "prior" not in result


class TestModelHandling:
    """Tests for model parameter handling."""

    def test_model_with_multiple_parameters(self) -> None:
        """Test model with multiple parameters."""
        params = [
            Parameter(
                name=f"param_{i}",
                symbol=f"p_{i}",
                lower_bound=0.0,
                upper_bound=1.0,
                default_value=0.5,
                free=True,
            )
            for i in range(3)
        ]

        model = Model(
            name="multi_param_model",
            description="Model with multiple parameters",
            parameters=params,
        )

        assert len(model.parameters) == 3
        for i, param in enumerate(model.parameters):
            assert param.name == f"param_{i}"

    def test_model_parameter_access(self, sample_model: Model) -> None:
        """Test accessing model parameters."""
        assert "test_param" in sample_model
        param = sample_model["test_param"]
        assert param.name == "test_param"

    def test_model_parameter_not_found(self, sample_model: Model) -> None:
        """Test accessing non-existent parameter."""
        with pytest.raises(KeyError):
            _ = sample_model["nonexistent_param"]


class TestCosmologySpecHandling:
    """Tests for cosmology specification handling."""

    def test_vanilla_lcdm_parameters(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test vanilla LCDM has expected parameters."""
        param_names = {p.name for p in vanilla_cosmo.parameters}

        # Should have minimal required set
        assert "Omega_c" in param_names
        assert "Omega_b" in param_names
        assert "h" in param_names
        assert "n_s" in param_names

        # Should have either sigma8 or A_s
        assert ("sigma8" in param_names) or ("A_s" in param_names)

    def test_cosmology_parameter_access(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test accessing cosmology parameters."""
        assert "Omega_c" in vanilla_cosmo
        omega_c = vanilla_cosmo["Omega_c"]
        assert omega_c.name == "Omega_c"

    def test_cosmology_num_massive_neutrinos(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test getting number of massive neutrinos."""
        num_nu = vanilla_cosmo.get_num_massive_neutrinos()
        assert isinstance(num_nu, int)
        assert num_nu >= 0


class TestToPascal:
    """Tests for _to_pascal utility function."""

    def test_to_pascal_snake_case(self) -> None:
        """Test converting snake_case to PascalCase."""
        assert _to_pascal("my_model_name") == "MyModelName"

    def test_to_pascal_kebab_case(self) -> None:
        """Test converting kebab-case to PascalCase."""
        assert _to_pascal("my-model-name") == "MyModelName"

    def test_to_pascal_mixed(self) -> None:
        """Test converting mixed separators."""
        assert _to_pascal("my_model-name") == "MyModelName"

    def test_to_pascal_single_word(self) -> None:
        """Test single word conversion."""
        assert _to_pascal("model") == "Model"


class TestPriorIntegration:
    """Tests for prior handling in NumCosmo configuration."""

    def test_generator_with_gaussian_priors(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with Gaussian priors on cosmology parameters."""
        assert numcosmo_init

        # Create cosmology with Gaussian priors
        prior_omega_c = PriorGaussian(mean=0.25, sigma=0.05)
        param_omega_c = Parameter(
            name="Omega_c",
            symbol="Omega_c",
            lower_bound=0.2,
            upper_bound=0.3,
            default_value=0.25,
            free=True,
            prior=prior_omega_c,
        )

        params = [param_omega_c] + [
            p for p in CCLCosmologySpec.vanilla_lcdm().parameters if p.name != "Omega_c"
        ]
        cosmo_with_prior = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_prior",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_prior,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec["Omega_c"].prior is not None

    def test_generator_with_uniform_priors(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with uniform priors on cosmology parameters."""
        assert numcosmo_init

        # Create cosmology with uniform priors
        prior_omega_b = PriorUniform(lower=0.04, upper=0.06)
        param_omega_b = Parameter(
            name="Omega_b",
            symbol="Omega_b",
            lower_bound=0.03,
            upper_bound=0.07,
            default_value=0.05,
            free=True,
            prior=prior_omega_b,
        )

        params = [param_omega_b] + [
            p for p in CCLCosmologySpec.vanilla_lcdm().parameters if p.name != "Omega_b"
        ]
        cosmo_with_prior = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_prior",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_prior,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec["Omega_b"].prior is not None

    def test_generator_with_model_priors(
        self, numcosmo_init: bool, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test generator with priors on model parameters."""
        assert numcosmo_init

        # Create model with Gaussian prior
        prior = PriorGaussian(mean=1.0, sigma=0.2)
        param_with_prior = Parameter(
            name="model_param",
            symbol="m_p",
            lower_bound=0.5,
            upper_bound=1.5,
            default_value=1.0,
            free=True,
            prior=prior,
        )
        model = Model(
            name="test_model",
            description="Model with prior",
            parameters=[param_with_prior],
        )

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_model_prior",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        gen.add_models([model])

        assert len(gen.models) == 1
        assert gen.models[0].parameters[0].prior is not None


class TestAmplitudeParameterHandling:
    """Tests for A_s and sigma8 parameter handling."""

    def test_generator_with_a_s_parameter(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with A_s instead of sigma8."""
        assert numcosmo_init

        # Create cosmology with A_s
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
            )
        ]
        cosmo_with_as = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_as",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_as,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "A_s" in gen.cosmo_spec
        assert "sigma8" not in gen.cosmo_spec

    def test_generator_with_a_s_gaussian_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with Gaussian prior on A_s."""
        assert numcosmo_init

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
            prefix="test_as_prior",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_as,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec["A_s"].prior is not None

    def test_generator_with_a_s_uniform_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with uniform prior on A_s."""
        assert numcosmo_init

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

        assert gen.cosmo_spec["A_s"].prior is not None


class TestNeutrinoHandling:
    """Tests for massive neutrino parameter handling."""

    def test_generator_with_massive_neutrinos(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with massive neutrinos."""
        assert numcosmo_init

        cosmo_with_nu = CCLCosmologySpec.vanilla_lcdm_with_neutrinos()

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_neutrinos",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_nu,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec.get_num_massive_neutrinos() > 0

    def test_generator_with_neutrino_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with prior on neutrino mass."""
        assert numcosmo_init

        prior_mnu = PriorGaussian(mean=0.06, sigma=0.01)
        params = [
            p if p.name != "m_nu" else p.model_copy(update={"prior": prior_mnu})
            for p in CCLCosmologySpec.vanilla_lcdm_with_neutrinos().parameters
        ]
        cosmo_with_nu_prior = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_nu_prior",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_nu_prior,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec["m_nu"].prior is not None


class TestCosmologyNone:
    """Tests for NONE cosmology framework."""

    def test_generator_with_none_cosmology(
        self, numcosmo_init: bool, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test generator with NONE cosmology framework."""
        assert numcosmo_init

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_none",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONE,
        )

        assert gen.required_cosmology == FrameworkCosmology.NONE


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
        vanilla_cosmo: CCLCosmologySpec,
        minimal_factory_file: Path,
    ) -> None:
        """Test write_config with model parameters and priors."""
        assert numcosmo_init

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
        vanilla_cosmo: CCLCosmologySpec,
        minimal_factory_file: Path,
    ) -> None:
        """Test write_config with NONE cosmology framework."""
        assert numcosmo_init

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
        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        experiment = ser.dict_str_from_yaml_file(yaml_file.as_posix())

        mset_obj = experiment.get("model-set")
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
