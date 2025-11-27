"""Unit tests for firecrown.app.cosmology command module.

Tests cosmology configuration generation, parameter parsing, and prior handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml

from firecrown.app.cosmology import (
    Cosmology,
    PriorWrapper,
    _parse_key_value,
    _parse_prior_dict,
    _parse_prior,
    Generate,
)
from firecrown.app.analysis import PriorUniform, PriorGaussian


class TestCosmologyEnum:
    """Tests for Cosmology enum."""

    def test_cosmology_values(self) -> None:
        """Test that Cosmology enum has expected values."""
        assert Cosmology.VANILLA_LCDM.value == "vanilla_lcdm"
        assert (
            Cosmology.VANILLA_LCDM_WITH_NEUTRINOS.value == "vanilla_lcdm_with_neutrinos"
        )

    def test_cosmology_count(self) -> None:
        """Test that Cosmology enum has expected number of values."""
        assert len(list(Cosmology)) == 2

    def test_cosmology_access_by_value(self) -> None:
        """Test accessing cosmology by value."""
        assert Cosmology("vanilla_lcdm") == Cosmology.VANILLA_LCDM
        assert (
            Cosmology("vanilla_lcdm_with_neutrinos")
            == Cosmology.VANILLA_LCDM_WITH_NEUTRINOS
        )

    def test_cosmology_string_representation(self) -> None:
        """Test string representation of cosmology enum."""
        assert str(Cosmology.VANILLA_LCDM) == "vanilla_lcdm"
        assert (
            str(Cosmology.VANILLA_LCDM_WITH_NEUTRINOS) == "vanilla_lcdm_with_neutrinos"
        )


class TestParseKeyValue:
    """Tests for _parse_key_value function."""

    def test_parse_key_value_with_value(self) -> None:
        """Test parsing key=value format."""
        key, value = _parse_key_value("Omega_c=0.26")
        assert key == "Omega_c"
        assert value == 0.26

    def test_parse_key_value_key_only(self) -> None:
        """Test parsing key-only format."""
        key, value = _parse_key_value("sigma8")
        assert key == "sigma8"
        assert value is None

    def test_parse_key_value_float_conversion(self) -> None:
        """Test float conversion of values."""
        key, value = _parse_key_value("h=0.6736")
        assert key == "h"
        assert isinstance(value, float)
        assert abs(value - 0.6736) < 1e-6

    def test_parse_key_value_negative_number(self) -> None:
        """Test parsing negative numeric values."""
        key, value = _parse_key_value("param=-0.5")
        assert key == "param"
        assert value == -0.5

    def test_parse_key_value_invalid_number(self) -> None:
        """Test error on invalid numeric value."""
        with pytest.raises(ValueError, match="must be a number"):
            _parse_key_value("Omega_c=invalid")

    def test_parse_key_value_with_multiple_equals(self) -> None:
        """Test parsing when value contains equals signs."""
        # Second part "2.5=test" fails float conversion
        with pytest.raises(ValueError):
            _parse_key_value("param=2.5=test")

    def test_parse_key_value_zero(self) -> None:
        """Test parsing zero value."""
        key, value = _parse_key_value("param=0")
        assert key == "param"
        assert value == 0.0


class TestParsePriorDict:
    """Tests for _parse_prior_dict function."""

    def test_parse_prior_dict_gaussian(self) -> None:
        """Test parsing Gaussian prior specification."""
        prior = _parse_prior_dict("mean=0.06,sigma=0.01")
        assert isinstance(prior, PriorGaussian)
        assert prior.mean == 0.06
        assert prior.sigma == 0.01

    def test_parse_prior_dict_uniform(self) -> None:
        """Test parsing uniform prior specification."""
        prior = _parse_prior_dict("lower=0.01,upper=0.1")
        assert isinstance(prior, PriorUniform)
        assert prior.lower == 0.01
        assert prior.upper == 0.1

    def test_parse_prior_dict_empty_string(self) -> None:
        """Test error on empty prior specification."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _parse_prior_dict("")

    def test_parse_prior_dict_invalid_format(self) -> None:
        """Test error on invalid format (no equals sign)."""
        with pytest.raises(ValueError, match="must be in format"):
            _parse_prior_dict("mean0.06")

    def test_parse_prior_dict_invalid_number(self) -> None:
        """Test error on invalid numeric value."""
        with pytest.raises(ValueError, match="must be a number"):
            _parse_prior_dict("mean=invalid,sigma=0.01")

    def test_parse_prior_dict_single_parameter(self) -> None:
        """Test parsing single prior parameter fails validation."""
        # Single parameter doesn't fully specify a prior
        with pytest.raises(Exception):  # Pydantic validation error
            _parse_prior_dict("mean=0.5")

    def test_parse_prior_dict_whitespace_handling(self) -> None:
        """Test that whitespace in values is preserved."""
        # Values are converted to float, so whitespace is stripped
        prior = _parse_prior_dict("mean= 0.06 ,sigma= 0.01 ")
        assert isinstance(prior, PriorGaussian)
        assert prior.mean == 0.06
        assert prior.sigma == 0.01


class TestParsePrior:
    """Tests for _parse_prior function."""

    def test_parse_prior_value_only(self) -> None:
        """Test parsing parameter with value only."""
        key, value, prior = _parse_prior("Omega_c=0.26")
        assert key == "Omega_c"
        assert value == 0.26
        assert prior is None

    def test_parse_prior_gaussian_with_value(self) -> None:
        """Test parsing parameter with value and Gaussian prior."""
        key, value, prior = _parse_prior("m_nu=0.06,mean=0.06,sigma=0.01")
        assert key == "m_nu"
        assert value == 0.06
        assert isinstance(prior, PriorGaussian)
        assert prior.mean == 0.06
        assert prior.sigma == 0.01

    def test_parse_prior_uniform_with_value(self) -> None:
        """Test parsing parameter with value and uniform prior."""
        key, value, prior = _parse_prior("Omega_c=0.26,lower=0.2,upper=0.3")
        assert key == "Omega_c"
        assert value == 0.26
        assert isinstance(prior, PriorUniform)
        assert prior.lower == 0.2
        assert prior.upper == 0.3

    def test_parse_prior_key_only_with_gaussian(self) -> None:
        """Test parsing parameter with prior but no value."""
        key, value, prior = _parse_prior("sigma8,mean=0.8,sigma=0.1")
        assert key == "sigma8"
        assert value is None
        assert isinstance(prior, PriorGaussian)
        assert prior.mean == 0.8
        assert prior.sigma == 0.1

    def test_parse_prior_key_only_with_uniform(self) -> None:
        """Test parsing parameter with uniform prior but no value."""
        key, value, prior = _parse_prior("Omega_b,lower=0.04,upper=0.05")
        assert key == "Omega_b"
        assert value is None
        assert isinstance(prior, PriorUniform)
        assert prior.lower == 0.04
        assert prior.upper == 0.05

    def test_parse_prior_key_only_no_prior(self) -> None:
        """Test error when parameter has no value and no prior."""
        with pytest.raises(ValueError, match="must have either"):
            _parse_prior("sigma8")

    def test_parse_prior_invalid_format(self) -> None:
        """Test error on invalid prior format."""
        with pytest.raises(ValueError):
            _parse_prior("invalid_format")

    def test_parse_prior_complex_value(self) -> None:
        """Test parsing parameter with scientific notation."""
        key, value, prior = _parse_prior("m_nu=1e-3,mean=1e-3,sigma=1e-4")
        assert key == "m_nu"
        assert isinstance(prior, PriorGaussian)
        assert isinstance(value, float)
        assert abs(value - 1e-3) < 1e-10
        assert abs(prior.mean - 1e-3) < 1e-10


class TestPriorWrapper:
    """Tests for PriorWrapper class."""

    def test_prior_wrapper_gaussian(self) -> None:
        """Test PriorWrapper with Gaussian prior."""
        wrapper = PriorWrapper.model_validate({"value": {"mean": 0.06, "sigma": 0.01}})
        assert isinstance(wrapper.value, PriorGaussian)

    def test_prior_wrapper_uniform(self) -> None:
        """Test PriorWrapper with uniform prior."""
        wrapper = PriorWrapper.model_validate({"value": {"lower": 0.01, "upper": 0.1}})
        assert isinstance(wrapper.value, PriorUniform)

    def test_prior_wrapper_extra_fields_rejected(self) -> None:
        """Test that PriorWrapper rejects extra fields."""
        with pytest.raises(Exception):  # Pydantic validation error
            PriorWrapper.model_validate({"value": {"mean": 0.06}, "extra": "field"})


class TestGenerateInitialization:
    """Tests for Generate command initialization."""

    @patch("firecrown.app.cosmology.logging.Logging.__init__", return_value=None)
    def test_generate_basic_attributes(
        self, _mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test that Generate stores basic attributes."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )
        assert gen.output_file == output_file
        assert gen.cosmology == Cosmology.VANILLA_LCDM

    @patch("firecrown.app.cosmology.logging.Logging.__init__", return_value=None)
    def test_generate_default_values(
        self, _mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test default values for optional parameters."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )
        assert gen.camb_halofit is None
        assert gen.parameter == []
        assert gen.exclude_defaults is False
        assert gen.print_output is False

    @patch("firecrown.app.cosmology.logging.Logging.__init__", return_value=None)
    def test_generate_with_parameters(
        self, _mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test Generate with parameter specifications."""
        output_file = tmp_path / "config.yaml"
        parameters = ["Omega_c=0.26", "sigma8,mean=0.8,sigma=0.1"]
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=parameters,
        )
        assert gen.parameter == parameters

    @patch("firecrown.app.cosmology.logging.Logging.__init__", return_value=None)
    def test_generate_with_camb_halofit(
        self, _mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test Generate with CAMB halofit option."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            camb_halofit="mead",
        )
        assert gen.camb_halofit == "mead"

    @patch("firecrown.app.cosmology.logging.Logging.__init__", return_value=None)
    def test_generate_with_print_output(
        self, _mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test Generate with print_output flag."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            print_output=True,
        )
        assert gen.print_output is True

    @patch("firecrown.app.cosmology.logging.Logging.__init__", return_value=None)
    def test_generate_with_exclude_defaults(
        self, _mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test Generate with exclude_defaults flag."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            exclude_defaults=True,
        )
        assert gen.exclude_defaults is True


class TestGenerateExecution:
    """Tests for Generate command execution (integration tests)."""

    def test_generate_vanilla_lcdm_creates_file(self, tmp_path: Path) -> None:
        """Test that Generate creates output file."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )
        gen.__post_init__()
        assert output_file.exists()

    def test_generate_vanilla_lcdm_valid_yaml(self, tmp_path: Path) -> None:
        """Test that generated file is valid YAML."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )
        gen.__post_init__()
        config = yaml.safe_load(output_file.read_text())
        assert isinstance(config, dict)
        assert "name" in config or len(config) > 0

    def test_generate_with_neutrinos(self, tmp_path: Path) -> None:
        """Test generating config with neutrinos."""
        output_file = tmp_path / "config_nu.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM_WITH_NEUTRINOS,
            parameter=[],
        )
        gen.__post_init__()
        assert output_file.exists()

    def test_generate_with_parameter_value(self, tmp_path: Path) -> None:
        """Test generating config with parameter value override."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.25"],
        )
        gen.__post_init__()
        config = yaml.safe_load(output_file.read_text())
        assert "Omega_c" in config or "parameters" in config

    def test_generate_with_multiple_parameters(self, tmp_path: Path) -> None:
        """Test generating config with multiple parameter overrides."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.25", "Omega_b=0.05"],
        )
        gen.__post_init__()
        assert output_file.exists()

    def test_generate_with_invalid_parameter(self, tmp_path: Path) -> None:
        """Test error when updating non-existent parameter."""
        output_file = tmp_path / "config.yaml"
        # The error occurs during __post_init__
        with pytest.raises(ValueError, match="Unknown parameter"):
            gen = Generate(
                output_file=output_file,
                cosmology=Cosmology.VANILLA_LCDM,
                parameter=["invalid_param=0.5"],
            )
            gen.__post_init__()

    def test_generate_mead_halofit(self, tmp_path: Path) -> None:
        """Test generating config with mead halofit."""
        output_file = tmp_path / "config_mead.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            camb_halofit="mead",
        )
        gen.__post_init__()
        assert output_file.exists()

    def test_generate_mead2020_feedback_halofit(self, tmp_path: Path) -> None:
        """Test generating config with mead2020_feedback halofit."""
        output_file = tmp_path / "config_mead2020.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            camb_halofit="mead2020_feedback",
        )
        gen.__post_init__()
        assert output_file.exists()

    def test_generate_exclude_defaults(self, tmp_path: Path) -> None:
        """Test generating config with exclude_defaults flag."""
        output_file = tmp_path / "config_no_defaults.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            exclude_defaults=True,
        )
        gen.__post_init__()
        assert output_file.exists()

    def test_generate_output_is_valid_python_dataclass_dict(
        self, tmp_path: Path
    ) -> None:
        """Test that generated YAML can be loaded as dict."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )
        gen.__post_init__()
        content = yaml.safe_load(output_file.read_text())
        assert isinstance(content, dict)

    def test_generate_creates_readable_file(self, tmp_path: Path) -> None:
        """Test that generated file is readable text."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )
        gen.__post_init__()
        content = output_file.read_text()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_generate_with_gaussian_prior(self, tmp_path: Path) -> None:
        """Test generating config with Gaussian prior."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.26,mean=0.26,sigma=0.02"],
        )
        gen.__post_init__()
        assert output_file.exists()

    def test_generate_with_uniform_prior(self, tmp_path: Path) -> None:
        """Test generating config with uniform prior."""
        output_file = tmp_path / "config.yaml"
        gen = Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.26,lower=0.2,upper=0.3"],
        )
        gen.__post_init__()
        assert output_file.exists()
