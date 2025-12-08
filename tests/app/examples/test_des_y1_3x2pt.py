"""Unit tests for firecrown.app.examples._des_y1_3x2pt module.

Tests ExampleDESY13x2pt example generator without executing likelihoods.
"""

from pathlib import Path
from unittest.mock import patch

from firecrown.likelihood import NamedParameters
from firecrown.app.examples._des_y1_3x2pt import (
    ExampleDESY13x2pt,
    DESY1FactoryType,
)
from firecrown.app.analysis import (
    Model,
    FrameworkCosmology,
    Frameworks,
)


class TestDESY1FactoryType:
    """Tests for DESY1FactoryType enum."""

    def test_factory_types_exist(self) -> None:
        """Test that all factory types are defined."""
        assert DESY1FactoryType.STANDARD.value == "standard"
        assert DESY1FactoryType.PT.value == "pt"
        assert DESY1FactoryType.TATT.value == "tatt"
        assert DESY1FactoryType.HMIA.value == "hmia"
        assert DESY1FactoryType.PK_MODIFIER.value == "pk_modifier"
        assert DESY1FactoryType.YAML_DEFAULT.value == "yaml_default"
        assert DESY1FactoryType.YAML_PURE_CCL.value == "yaml_pure_ccl"
        assert DESY1FactoryType.YAML_MU_SIGMA.value == "yaml_mu_sigma"


class TestExampleDESY13x2pt:
    """Tests for ExampleDESY13x2pt class."""

    def test_class_attributes(self) -> None:
        """Test that class has required attributes."""
        assert hasattr(ExampleDESY13x2pt, "description")
        assert hasattr(ExampleDESY13x2pt, "data_url")
        assert "DES Y1" in ExampleDESY13x2pt.description
        assert "3x2pt" in ExampleDESY13x2pt.description
        assert "github.com" in ExampleDESY13x2pt.data_url

    def test_default_parameters(self, tmp_path: Path) -> None:
        """Test that default parameters are set correctly."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                target_framework=Frameworks.COSMOSIS,
            )

        assert builder.prefix == "des_y1_3x2pt"
        assert builder.factory_type == DESY1FactoryType.STANDARD

    def test_custom_factory_type(self, tmp_path: Path) -> None:
        """Test that custom factory type is applied."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                factory_type=DESY1FactoryType.PT,
                target_framework=Frameworks.COSMOSIS,
            )

        assert builder.factory_type == DESY1FactoryType.PT

    def test_generate_sacc_downloads_file(self, tmp_path: Path) -> None:
        """Test that generate_sacc calls download_from_url."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                target_framework=Frameworks.COSMOSIS,
            )

        with patch(
            "firecrown.app.examples._des_y1_3x2pt.download_from_url"
        ) as mock_download:
            result = builder.generate_sacc(tmp_path)

            mock_download.assert_called_once()
            assert result == tmp_path / "test_des.sacc"
            call_args = mock_download.call_args[0]
            assert ExampleDESY13x2pt.data_url == call_args[0]

    def test_generate_factory_standard(self, tmp_path: Path) -> None:
        """Test generate_factory with standard factory type."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.STANDARD,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        with patch("firecrown.app.examples._des_y1_3x2pt.copy_template") as mock_copy:
            result = builder.generate_factory(tmp_path, sacc_path)

            mock_copy.assert_called_once()
            assert result == tmp_path / "test_des_factory.py"

    def test_generate_factory_pt(self, tmp_path: Path) -> None:
        """Test generate_factory with PT factory type."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.PT,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        with patch("firecrown.app.examples._des_y1_3x2pt.copy_template") as mock_copy:
            result = builder.generate_factory(tmp_path, sacc_path)

            mock_copy.assert_called_once()
            assert result == tmp_path / "test_des_factory.py"

    def test_generate_factory_tatt(self, tmp_path: Path) -> None:
        """Test generate_factory with TATT factory type."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.TATT,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        with patch("firecrown.app.examples._des_y1_3x2pt.copy_template") as mock_copy:
            result = builder.generate_factory(tmp_path, sacc_path)

            mock_copy.assert_called_once()
            assert result == tmp_path / "test_des_factory.py"

    def test_generate_factory_hmia(self, tmp_path: Path) -> None:
        """Test generate_factory with HMIA factory type."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.HMIA,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        with patch("firecrown.app.examples._des_y1_3x2pt.copy_template") as mock_copy:
            result = builder.generate_factory(tmp_path, sacc_path)

            mock_copy.assert_called_once()
            assert result == tmp_path / "test_des_factory.py"

    def test_generate_factory_pk_modifier(self, tmp_path: Path) -> None:
        """Test generate_factory with PK_MODIFIER factory type."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.PK_MODIFIER,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        with patch("firecrown.app.examples._des_y1_3x2pt.copy_template") as mock_copy:
            result = builder.generate_factory(tmp_path, sacc_path)

            mock_copy.assert_called_once()
            assert result == tmp_path / "test_des_factory.py"

    def test_generate_factory_yaml_default(self, tmp_path: Path) -> None:
        """Test generate_factory with YAML_DEFAULT factory type."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.YAML_DEFAULT,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        result = builder.generate_factory(tmp_path, sacc_path)

        # YAML factory returns a string, not a Path
        assert isinstance(result, str)
        assert "firecrown.likelihood.factories" in result

    def test_generate_factory_yaml_pure_ccl(self, tmp_path: Path) -> None:
        """Test generate_factory with YAML_PURE_CCL factory type."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.YAML_PURE_CCL,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        result = builder.generate_factory(tmp_path, sacc_path)

        assert isinstance(result, str)
        assert "firecrown.likelihood.factories" in result

    def test_generate_factory_yaml_mu_sigma(self, tmp_path: Path) -> None:
        """Test generate_factory with YAML_MU_SIGMA factory type."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.YAML_MU_SIGMA,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        result = builder.generate_factory(tmp_path, sacc_path)

        assert isinstance(result, str)
        assert "firecrown.likelihood.factories" in result

    def test_get_build_parameters(self, tmp_path: Path) -> None:
        """Test that get_build_parameters returns NamedParameters."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        params = builder.get_build_parameters(sacc_path)

        assert isinstance(params, NamedParameters)
        assert "sacc_file" in params.convert_to_basic_dict()

    def test_get_models_standard(self, tmp_path: Path) -> None:
        """Test that get_models returns model parameters for standard factory."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.STANDARD,
                target_framework=Frameworks.COSMOSIS,
            )

        models = builder.get_models()

        assert isinstance(models, list)
        assert len(models) == 1
        assert isinstance(models[0], Model)
        # Should have lens and source bias parameters
        param_names = {p.name for p in models[0].parameters}
        assert any("lens" in name for name in param_names)
        assert any("src" in name for name in param_names)

    def test_get_models_pt(self, tmp_path: Path) -> None:
        """Test that get_models returns PT-specific parameters."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.PT,
                target_framework=Frameworks.COSMOSIS,
            )

        models = builder.get_models()

        assert isinstance(models, list)
        assert len(models) == 1
        # PT should have lens bias parameters
        param_names = {p.name for p in models[0].parameters}
        assert any("lens" in name and "bias" in name for name in param_names)

    def test_get_models_yaml(self, tmp_path: Path) -> None:
        """Test that get_models returns empty list for YAML factories."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.YAML_DEFAULT,
                target_framework=Frameworks.COSMOSIS,
            )

        models = builder.get_models()

        # YAML factories DO define models in Python (for parameter sampling)
        assert isinstance(models, list)
        assert len(models) >= 1
        # Should have bias and photo-z parameters
        if len(models) > 0:
            param_names = {p.name for p in models[0].parameters}
            assert any("bias" in name or "delta_z" in name for name in param_names)

    def test_required_cosmology(self, tmp_path: Path) -> None:
        """Test that required_cosmology returns NONLINEAR."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                target_framework=Frameworks.COSMOSIS,
            )

        result = builder.required_cosmology()
        assert result == FrameworkCosmology.NONLINEAR

    def test_get_options_desc(self, tmp_path: Path) -> None:
        """Test that get_options_desc returns factory type info."""
        with patch.object(ExampleDESY13x2pt, "_proceed_generation"):
            builder = ExampleDESY13x2pt(
                output_path=tmp_path,
                prefix="test_des",
                factory_type=DESY1FactoryType.TATT,
                target_framework=Frameworks.COSMOSIS,
            )

        options = builder.get_options_desc()

        assert isinstance(options, list)
        assert len(options) > 0
        # Should have factory type info
        option_names = [name for name, _ in options]
        assert any("factory" in name.lower() for name in option_names)
