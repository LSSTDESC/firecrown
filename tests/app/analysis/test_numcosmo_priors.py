"""Unit tests for NumCosmo prior and model handling.

Tests for prior handling (Gaussian and Uniform priors), model parameter handling,
and prior integration with the generator in firecrown.app.analysis._numcosmo module.
"""

from pathlib import Path
import pytest

from numcosmo_py import Ncm
from firecrown.app.analysis._numcosmo import (
    NumCosmoConfigGenerator,
    _param_to_nc_dict,
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
