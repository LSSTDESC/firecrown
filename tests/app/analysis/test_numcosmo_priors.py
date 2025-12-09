"""Unit tests for NumCosmo prior and model handling.

Tests for prior handling (Gaussian and Uniform priors), model parameter handling,
and prior integration with the generator in firecrown.app.analysis._numcosmo module.
"""

from pathlib import Path
import pytest

from numcosmo_py import Ncm
from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._numcosmo import (
    ConfigOptions,
    NumCosmoConfigGenerator,
    _param_to_nc_dict,
    _setup_models,
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


class TestPriorIntegration:
    """Tests for prior handling in NumCosmo configuration."""

    def test_generator_with_gaussian_priors(
        self, numcosmo_init: None, tmp_path: Path
    ) -> None:
        """Test generator with Gaussian priors on cosmology parameters."""
        # Create cosmology with Gaussian priors
        assert numcosmo_init
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
        self, numcosmo_init: None, tmp_path: Path
    ) -> None:
        """Test generator with uniform priors on cosmology parameters."""
        # Create cosmology with uniform priors
        assert numcosmo_init
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
        self, numcosmo_init: None, tmp_path: Path
    ) -> None:
        """Test generator with priors on model parameters."""
        assert numcosmo_init
        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()

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


class TestSetupModels:
    """Tests for _setup_models function."""

    def test_setup_models_with_gaussian_priors(self, numcosmo_init: bool) -> None:
        """Test _setup_models with Gaussian priors on model parameters.

        This test verifies that _setup_models correctly:
        1. Creates and registers model builders
        2. Adds models to the model set
        3. Adds Gaussian priors to the priors list for parameters with priors
        """
        assert numcosmo_init

        # Create model with Gaussian prior
        prior_gauss = PriorGaussian(mean=1.0, sigma=0.2)
        param_with_prior = Parameter(
            name="bias",
            symbol="b",
            lower_bound=0.5,
            upper_bound=2.0,
            default_value=1.0,
            free=True,
            prior=prior_gauss,
        )
        model = Model(
            name="bias_model",
            description="Galaxy bias model",
            parameters=[param_with_prior],
        )

        # Create config options
        config_opts = ConfigOptions(
            output_path=Path("/tmp"),
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[model],
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo model set and priors list
        mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _setup_models
        model_builders = _setup_models(config_opts, mset, priors)

        # Verify model builders were created
        assert model_builders is not None
        assert isinstance(model_builders, Ncm.ObjDictStr)

        # Verify model was added to model set
        assert mset.nmodels() == 1

        # Verify prior was added
        assert len(priors) == initial_priors_count + 1
        assert isinstance(priors[-1], Ncm.PriorGauss)

    def test_setup_models_with_uniform_priors(self, numcosmo_init: bool) -> None:
        """Test _setup_models with uniform priors on model parameters.

        This test verifies that _setup_models correctly adds uniform priors
        to the priors list for parameters with uniform prior specifications.
        """
        assert numcosmo_init

        # Create model with uniform prior
        prior_uniform = PriorUniform(lower=0.8, upper=1.5)
        param_with_prior = Parameter(
            name="amplitude",
            symbol="A",
            lower_bound=0.0,
            upper_bound=3.0,
            default_value=1.0,
            free=True,
            prior=prior_uniform,
        )
        model = Model(
            name="amplitude_model",
            description="Amplitude model",
            parameters=[param_with_prior],
        )

        # Create config options
        config_opts = ConfigOptions(
            output_path=Path("/tmp"),
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[model],
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo model set and priors list
        mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _setup_models
        _ = _setup_models(config_opts, mset, priors)

        # Verify model was added
        assert mset.nmodels() == 1

        # Verify prior was added
        assert len(priors) == initial_priors_count + 1
        assert isinstance(priors[-1], Ncm.PriorFlat)

    def test_setup_models_without_priors(self, numcosmo_init: bool) -> None:
        """Test _setup_models with parameters that have no priors.

        This test verifies that _setup_models does not add any priors
        when model parameters have no prior specifications.
        """
        assert numcosmo_init

        # Create model without priors
        param_no_prior = Parameter(
            name="slope",
            symbol="s",
            lower_bound=-1.0,
            upper_bound=1.0,
            default_value=0.0,
            free=True,
            prior=None,  # No prior
        )
        model = Model(
            name="slope_model",
            description="Slope model without prior",
            parameters=[param_no_prior],
        )

        # Create config options
        config_opts = ConfigOptions(
            output_path=Path("/tmp"),
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[model],
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo model set and priors list
        mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _setup_models
        _ = _setup_models(config_opts, mset, priors)

        # Verify model was added
        assert mset.nmodels() == 1

        # Verify no prior was added
        assert len(priors) == initial_priors_count

    def test_setup_models_mixed_priors(self, numcosmo_init: bool) -> None:
        """Test _setup_models with multiple parameters, some with priors and some without.

        This test verifies that _setup_models correctly handles a model with
        multiple parameters where only some have priors.
        """
        assert numcosmo_init

        # Create model with mixed priors
        param_with_gauss = Parameter(
            name="param1",
            symbol="p1",
            lower_bound=0.0,
            upper_bound=2.0,
            default_value=1.0,
            free=True,
            prior=PriorGaussian(mean=1.0, sigma=0.1),
        )
        param_without_prior = Parameter(
            name="param2",
            symbol="p2",
            lower_bound=0.0,
            upper_bound=2.0,
            default_value=1.0,
            free=True,
            prior=None,
        )
        param_with_uniform = Parameter(
            name="param3",
            symbol="p3",
            lower_bound=0.0,
            upper_bound=2.0,
            default_value=1.0,
            free=True,
            prior=PriorUniform(lower=0.5, upper=1.5),
        )

        model = Model(
            name="mixed_model",
            description="Model with mixed priors",
            parameters=[param_with_gauss, param_without_prior, param_with_uniform],
        )

        # Create config options
        config_opts = ConfigOptions(
            output_path=Path("/tmp"),
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[model],
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo model set and priors list
        mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _setup_models
        _ = _setup_models(config_opts, mset, priors)

        # Verify model was added
        assert mset.nmodels() == 1

        # Verify exactly 2 priors were added (for param1 and param3, not param2)
        assert len(priors) == initial_priors_count + 2
        assert isinstance(priors[-2], Ncm.PriorGauss)
        assert isinstance(priors[-1], Ncm.PriorFlat)

    def test_setup_models_multiple_models(self, numcosmo_init: bool) -> None:
        """Test _setup_models with multiple models.

        This test verifies that _setup_models correctly handles multiple models,
        each with their own parameters and priors.
        """
        assert numcosmo_init

        # Create first model with Gaussian prior
        model1 = Model(
            name="model_one",
            description="First model",
            parameters=[
                Parameter(
                    name="param_a",
                    symbol="a",
                    lower_bound=0.0,
                    upper_bound=2.0,
                    default_value=1.0,
                    free=True,
                    prior=PriorGaussian(mean=1.0, sigma=0.2),
                )
            ],
        )

        # Create second model with uniform prior
        model2 = Model(
            name="model_two",
            description="Second model",
            parameters=[
                Parameter(
                    name="param_b",
                    symbol="b",
                    lower_bound=0.0,
                    upper_bound=2.0,
                    default_value=1.0,
                    free=True,
                    prior=PriorUniform(lower=0.5, upper=1.5),
                )
            ],
        )

        # Create config options with both models
        config_opts = ConfigOptions(
            output_path=Path("/tmp"),
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[model1, model2],
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo model set and priors list
        mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _setup_models
        _ = _setup_models(config_opts, mset, priors)

        # Verify both models were added
        assert mset.nmodels() == 2

        # Verify both priors were added
        assert len(priors) == initial_priors_count + 2
        assert isinstance(priors[-2], Ncm.PriorGauss)
        assert isinstance(priors[-1], Ncm.PriorFlat)

    def test_setup_models_no_models(self, numcosmo_init: bool) -> None:
        """Test _setup_models with no models.

        This test verifies that _setup_models handles the case where
        no models are provided in the configuration.
        """
        assert numcosmo_init

        # Create config options with no models
        config_opts = ConfigOptions(
            output_path=Path("/tmp"),
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],  # No models
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo model set and priors list
        mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _setup_models
        _ = _setup_models(config_opts, mset, priors)

        # Verify no models were added
        assert mset.nmodels() == 0

        # Verify no priors were added
        assert len(priors) == initial_priors_count
