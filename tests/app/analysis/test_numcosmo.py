"""Unit tests for firecrown.app.analysis._numcosmo module.

Tests NumCosmo configuration generation and parameter handling.
"""

from pathlib import Path
import pytest

from numcosmo_py import Ncm

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
        self, numcosmo_init: bool, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
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

        # Create a minimal factory file
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
        "NZ",
        "lens0",
        np.array([0.1, 0.2, 0.3]),
        np.array([0.0, 1.0, 0.0]),
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

        gen.factory_source = factory_file
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
