"""Unit tests for NumCosmo utility functions.

Tests for utility functions in firecrown.app.analysis._numcosmo module,
including parameter mapping, name conversion, and parameter dictionary creation.
"""

import pytest

from firecrown.app.analysis._numcosmo import (
    NAME_MAP,
    _param_to_nc_dict,
    _to_pascal,
)
from firecrown.app.analysis._types import Parameter


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
