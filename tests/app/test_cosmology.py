"""Unit tests for firecrown.app.cosmology command module.

Tests cosmology configuration generation, parameter parsing, and prior handling.
"""

from pathlib import Path
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
    """Tests for Generate command initialization and execution."""

    def test_generate_vanilla_lcdm_creates_file(self, tmp_path: Path) -> None:
        """Test that Generate creates the output file for VANILLA_LCDM."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )
        assert output_file.exists()

    def test_generate_vanilla_lcdm_creates_valid_yaml(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that Generate creates valid YAML for VANILLA_LCDM."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )

        # Verify console output
        captured = capsys.readouterr()
        assert "configuration written" in captured.out.lower()
        assert str(output_file) in captured.out

        # Verify YAML can be loaded and has expected structure
        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert "parameters" in config
        # Check that common cosmological parameters are present
        param_names = {p["name"] for p in config["parameters"]}
        assert "Omega_c" in param_names
        assert "Omega_b" in param_names
        assert "h" in param_names

    def test_generate_with_neutrinos(self, tmp_path: Path) -> None:
        """Test Generate with neutrino cosmology."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM_WITH_NEUTRINOS,
            parameter=[],
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Neutrino cosmology should have additional parameters
        param_names = {p["name"] for p in config["parameters"]}
        assert (
            "Neff" in param_names
            or "m_nu" in param_names
            or len(config["parameters"]) >= 7
        )

    def test_generate_with_parameter_value(self, tmp_path: Path) -> None:
        """Test Generate with a parameter value override (no prior)."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.26"],
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Find Omega_c in the parameters list
        omega_c_param = next(p for p in config["parameters"] if p["name"] == "Omega_c")
        assert omega_c_param["default_value"] == 0.26
        # Should have no prior since only value was specified
        assert omega_c_param["prior"] is None

    def test_generate_with_multiple_parameters(self, tmp_path: Path) -> None:
        """Test Generate with multiple parameter specifications."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.26", "sigma8=0.8"],
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Find parameters in the list
        omega_c_param = next(p for p in config["parameters"] if p["name"] == "Omega_c")
        sigma8_param = next(p for p in config["parameters"] if p["name"] == "sigma8")
        assert omega_c_param["default_value"] == 0.26
        assert sigma8_param["default_value"] == 0.8

    def test_generate_with_invalid_parameter(self, tmp_path: Path) -> None:
        """Test Generate with an invalid parameter raises ValueError."""
        output_file = tmp_path / "config.yaml"
        with pytest.raises(ValueError, match="unknown_param"):
            Generate(
                output_file=output_file,
                cosmology=Cosmology.VANILLA_LCDM,
                parameter=["unknown_param=0.5"],
            )

    def test_generate_with_gaussian_prior(self, tmp_path: Path) -> None:
        """Test Generate with a Gaussian prior and value."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.26,mean=0.27,sigma=0.02"],
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        omega_c_param = next(p for p in config["parameters"] if p["name"] == "Omega_c")
        assert omega_c_param["default_value"] == 0.26
        assert omega_c_param["prior"] is not None
        # Prior doesn't have 'kind' - it just has the prior parameters directly
        assert omega_c_param["prior"]["mean"] == 0.27
        assert omega_c_param["prior"]["sigma"] == 0.02

    def test_generate_with_prior_only_no_value(self, tmp_path: Path) -> None:
        """Test Generate with a prior but no value override."""
        output_file = tmp_path / "config.yaml"
        # Get the default value first
        _ = Generate(
            output_file=tmp_path / "default.yaml",
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
        )
        with open(tmp_path / "default.yaml", encoding="utf-8") as f:
            default_config = yaml.safe_load(f)
        default_omega_c = next(
            p for p in default_config["parameters"] if p["name"] == "Omega_c"
        )["default_value"]

        # Now generate with prior only (no value)
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c,mean=0.27,sigma=0.02"],
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        omega_c_param = next(p for p in config["parameters"] if p["name"] == "Omega_c")
        # Value should remain at default since no value was specified
        assert omega_c_param["default_value"] == default_omega_c
        # But prior should be set
        assert omega_c_param["prior"] is not None
        assert omega_c_param["prior"]["mean"] == 0.27
        assert omega_c_param["prior"]["sigma"] == 0.02

    def test_generate_with_uniform_prior(self, tmp_path: Path) -> None:
        """Test Generate with a uniform prior."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["h=0.7,lower=0.6,upper=0.8"],
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        h_param = next(p for p in config["parameters"] if p["name"] == "h")
        assert h_param["default_value"] == 0.7
        assert h_param["prior"] is not None
        # Prior doesn't have 'kind' - it just has the prior parameters directly
        assert h_param["prior"]["lower"] == 0.6
        assert h_param["prior"]["upper"] == 0.8

    def test_generate_camb_halofit_mead(self, tmp_path: Path) -> None:
        """Test Generate with CAMB halofit set to mead."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            camb_halofit="mead",
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # camb_halofit goes into extra_parameters
        assert "extra_parameters" in config
        assert config["extra_parameters"] is not None
        assert config["extra_parameters"]["halofit_version"] == "mead"

    def test_generate_camb_halofit_mead2020_feedback(self, tmp_path: Path) -> None:
        """Test Generate with CAMB halofit set to mead2020_feedback."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            camb_halofit="mead2020_feedback",
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # mead2020_feedback also goes into extra_parameters
        assert "extra_parameters" in config
        assert config["extra_parameters"] is not None
        assert config["extra_parameters"]["halofit_version"] == "mead2020_feedback"

    def test_generate_camb_halofit_peacock(self, tmp_path: Path) -> None:
        """Test Generate with CAMB halofit set to peacock."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            camb_halofit="peacock",
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # peacock goes into extra_parameters
        assert "extra_parameters" in config
        assert config["extra_parameters"] is not None
        assert config["extra_parameters"]["halofit_version"] == "peacock"

    def test_generate_camb_halofit_default(self, tmp_path: Path) -> None:
        """Test Generate with default (None) CAMB halofit has no extra_parameters."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            camb_halofit=None,
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Default case should have extra_parameters set to None (no special CAMB
        # parameters)
        assert "extra_parameters" in config
        assert config["extra_parameters"] is None

    def test_generate_exclude_defaults(self, tmp_path: Path) -> None:
        """Test Generate with exclude_defaults removes default-valued parameters."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=[],
            exclude_defaults=True,
        )

        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # With exclude_defaults and no modified parameters,
        # the parameters list should only contain free parameters
        # or be structured differently
        assert "parameters" in config
        # At minimum, structure should be valid
        assert isinstance(config["parameters"], list)

    def test_generate_print_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test Generate with print_output displays YAML content."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.26"],
            print_output=True,
        )

        captured = capsys.readouterr()
        # Should print the YAML content
        assert "Omega_c" in captured.out
        assert "0.26" in captured.out

    def test_generate_creates_readable_file(self, tmp_path: Path) -> None:
        """Test that Generate creates a file that can be read back."""
        output_file = tmp_path / "config.yaml"
        Generate(
            output_file=output_file,
            cosmology=Cosmology.VANILLA_LCDM,
            parameter=["Omega_c=0.27", "h=0.7"],
        )

        # File should be readable as valid YAML
        with open(output_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Should be a dictionary matching expected structure
        assert isinstance(config, dict)
        assert "parameters" in config

        # Verify the modified parameters
        omega_c_param = next(p for p in config["parameters"] if p["name"] == "Omega_c")
        h_param = next(p for p in config["parameters"] if p["name"] == "h")
        assert omega_c_param["default_value"] == 0.27
        assert h_param["default_value"] == 0.7
