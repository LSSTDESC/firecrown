"""Unit tests for firecrown.app.analysis types module.

Tests type definitions, enums, and data models for framework configuration.
"""

import pytest
from pydantic import ValidationError

from firecrown.app.analysis import (
    Frameworks,
    FrameworkCosmology,
    Parameter,
    Model,
    PriorUniform,
    PriorGaussian,
    COSMO_DESC,
    CCL_COSMOLOGY_MINIMAL_SET,
)


class TestFrameworksEnum:
    """Tests for Frameworks enum."""

    def test_frameworks_values(self) -> None:
        """Test that Frameworks enum has expected values."""
        assert Frameworks.COBAYA.value == "cobaya"
        assert Frameworks.COSMOSIS.value == "cosmosis"
        assert Frameworks.NUMCOSMO.value == "numcosmo"

    def test_frameworks_count(self) -> None:
        """Test that Frameworks enum has exactly 3 frameworks."""
        frameworks = list(Frameworks)
        assert len(frameworks) == 3

    def test_frameworks_creation_from_string(self) -> None:
        """Test creating Frameworks from string."""
        f = Frameworks("cobaya")
        assert f == Frameworks.COBAYA


class TestFrameworkCosmologyEnum:
    """Tests for FrameworkCosmology enum."""

    def test_framework_cosmology_values(self) -> None:
        """Test FrameworkCosmology enum values."""
        assert FrameworkCosmology.NONE.value == "none"
        assert FrameworkCosmology.BACKGROUND.value == "background"
        assert FrameworkCosmology.LINEAR.value == "linear"
        assert FrameworkCosmology.NONLINEAR.value == "nonlinear"

    def test_framework_cosmology_count(self) -> None:
        """Test that FrameworkCosmology has exactly 4 levels."""
        levels = list(FrameworkCosmology)
        assert len(levels) == 4


class TestPriorUniform:
    """Tests for PriorUniform."""

    def test_prior_uniform_creation(self) -> None:
        """Test creating a valid uniform prior."""
        prior = PriorUniform(lower=0.1, upper=0.9)
        assert prior.lower == 0.1
        assert prior.upper == 0.9

    def test_prior_uniform_invalid_bounds(self) -> None:
        """Test that uniform prior validates bounds."""
        with pytest.raises(ValidationError):
            PriorUniform(lower=0.9, upper=0.1)

    def test_prior_uniform_equal_bounds(self) -> None:
        """Test that uniform prior rejects equal bounds."""
        with pytest.raises(ValidationError):
            PriorUniform(lower=0.5, upper=0.5)


class TestPriorGaussian:
    """Tests for PriorGaussian."""

    def test_prior_gaussian_creation(self) -> None:
        """Test creating a valid Gaussian prior."""
        prior = PriorGaussian(mean=0.5, sigma=0.1)
        assert prior.mean == 0.5
        assert prior.sigma == 0.1

    def test_prior_gaussian_zero_sigma(self) -> None:
        """Test that Gaussian prior rejects zero sigma."""
        with pytest.raises(ValidationError):
            PriorGaussian(mean=0.5, sigma=0.0)

    def test_prior_gaussian_negative_sigma(self) -> None:
        """Test that Gaussian prior rejects negative sigma."""
        with pytest.raises(ValidationError):
            PriorGaussian(mean=0.5, sigma=-0.1)

    def test_prior_gaussian_negative_mean(self) -> None:
        """Test Gaussian prior with negative mean."""
        prior = PriorGaussian(mean=-0.5, sigma=0.1)
        assert prior.mean == -0.5


class TestParameter:
    """Tests for Parameter."""

    def test_parameter_creation(self) -> None:
        """Test creating a valid parameter."""
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
        )
        assert param.name == "Omega_c"
        assert param.symbol == r"\Omega_c"
        assert param.lower_bound == 0.1
        assert param.upper_bound == 0.5
        assert param.default_value == 0.3
        assert param.free is True

    def test_parameter_with_uniform_prior(self) -> None:
        """Test parameter with uniform prior."""
        prior = PriorUniform(lower=0.1, upper=0.5)
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
            prior=prior,
        )
        assert param is not None
        assert param.prior is not None
        assert isinstance(param.prior, PriorUniform)
        assert param.prior.lower == 0.1  # pylint: disable=no-member

    def test_parameter_with_gaussian_prior(self) -> None:
        """Test parameter with Gaussian prior."""
        prior = PriorGaussian(mean=0.3, sigma=0.05)
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
            prior=prior,
        )
        assert param.prior is not None
        assert isinstance(param.prior, PriorGaussian)

    def test_parameter_invalid_bounds(self) -> None:
        """Test that parameter validates bounds."""
        with pytest.raises(ValidationError):
            Parameter(
                name="test",
                symbol="t",
                lower_bound=0.5,
                upper_bound=0.1,
                default_value=0.3,
                free=True,
            )

    def test_parameter_auto_scale(self) -> None:
        """Test parameter auto-scale calculation."""
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
        )
        # Scale should be auto-calculated from bounds
        assert param.scale > 0

    def test_parameter_from_tuple(self) -> None:
        """Test creating parameter from tuple."""
        param = Parameter.from_tuple(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
        )
        assert param.name == "Omega_c"
        assert param.lower_bound == 0.1


class TestModel:
    """Tests for Model."""

    def test_model_creation(self) -> None:
        """Test creating a valid model."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
            Parameter.from_tuple("h", "h", 0.6, 0.8, 0.7, False),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        assert model.name == "cosmology"
        assert len(model.parameters) == 2

    def test_model_parameter_access(self) -> None:
        """Test accessing model parameters."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        param = model["Omega_c"]
        assert param.name == "Omega_c"

    def test_model_parameter_not_found(self) -> None:
        """Test KeyError when parameter not found."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        with pytest.raises(KeyError):
            _ = model["nonexistent"]

    def test_model_contains(self) -> None:
        """Test checking if parameter exists in model."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        assert "Omega_c" in model
        assert "nonexistent" not in model

    def test_model_duplicate_parameter_names(self) -> None:
        """Test that model rejects duplicate parameter names."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
            Parameter.from_tuple("Omega_c", "h", 0.6, 0.8, 0.7, False),
        ]
        with pytest.raises(ValueError, match="Duplicate parameter name"):
            Model(
                name="cosmology",
                description="Cosmological parameters",
                parameters=params,
            )

    def test_model_has_priors_true(self) -> None:
        """Test has_priors when priors are present."""
        prior = PriorUniform(lower=0.1, upper=0.5)
        params = [
            Parameter(
                name="Omega_c",
                symbol=r"\Omega_c",
                lower_bound=0.1,
                upper_bound=0.5,
                default_value=0.3,
                free=True,
                prior=prior,
            ),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        assert model.has_priors() is True

    def test_model_has_priors_false(self) -> None:
        """Test has_priors when no priors are present."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        assert model.has_priors() is False


class TestCosmoDesc:
    """Tests for COSMO_DESC dictionary."""

    def test_cosmo_desc_contains_minimal_set(self) -> None:
        """Test that COSMO_DESC contains all minimal cosmology parameters."""
        for param_name in CCL_COSMOLOGY_MINIMAL_SET:
            assert param_name in COSMO_DESC
            assert isinstance(COSMO_DESC[param_name], Parameter)

    def test_cosmo_desc_parameter_properties(self) -> None:
        """Test properties of COSMO_DESC parameters."""
        omega_c = COSMO_DESC["Omega_c"]
        assert omega_c.name == "Omega_c"
        assert omega_c.symbol == r"\Omega_c"
        assert omega_c.lower_bound < omega_c.upper_bound
        assert omega_c.lower_bound <= omega_c.default_value <= omega_c.upper_bound
