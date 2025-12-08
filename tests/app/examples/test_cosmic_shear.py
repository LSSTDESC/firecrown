"""Unit tests for firecrown.app.examples._cosmic_shear module.

Tests ExampleCosmicShear example generator without executing likelihoods.
"""

from pathlib import Path
from unittest.mock import patch

from firecrown.likelihood import NamedParameters
from firecrown.app.examples._cosmic_shear import ExampleCosmicShear
from firecrown.app.analysis import (
    FrameworkCosmology,
    Frameworks,
)


class TestExampleCosmicShear:
    """Tests for ExampleCosmicShear class."""

    def test_class_attributes(self) -> None:
        """Test that class has required attributes."""
        assert hasattr(ExampleCosmicShear, "description")
        assert (
            "Cosmic shear" in ExampleCosmicShear.description
            or "lensing" in ExampleCosmicShear.description
        )

    def test_default_parameters(self, tmp_path: Path) -> None:
        """Test that default parameters are set correctly."""
        with patch.object(ExampleCosmicShear, "_proceed_generation"):
            builder = ExampleCosmicShear(
                output_path=tmp_path,
                target_framework=Frameworks.COSMOSIS,
            )

        assert builder.prefix == "cosmic_shear"
        assert builder.seed == 42
        assert builder.n_bins == 2
        assert builder.z_max == 2.0

    def test_custom_parameters(self, tmp_path: Path) -> None:
        """Test that custom parameters are applied."""
        with patch.object(ExampleCosmicShear, "_proceed_generation"):
            builder = ExampleCosmicShear(
                output_path=tmp_path,
                prefix="custom_cs",
                seed=123,
                n_bins=3,
                z_max=3.0,
                target_framework=Frameworks.COBAYA,
            )

        assert builder.prefix == "custom_cs"
        assert builder.seed == 123
        assert builder.n_bins == 3
        assert builder.z_max == 3.0

    def test_generate_sacc_returns_path(self, tmp_path: Path) -> None:
        """Test that generate_sacc returns expected path."""
        with patch.object(ExampleCosmicShear, "_proceed_generation"):
            builder = ExampleCosmicShear(
                output_path=tmp_path,
                prefix="test_cs",
                target_framework=Frameworks.COSMOSIS,
            )

        # Mock sacc.Sacc to avoid actual computation
        with patch("firecrown.app.examples._cosmic_shear.sacc.Sacc"):
            with patch("firecrown.app.examples._cosmic_shear.pyccl"):
                result = builder.generate_sacc(tmp_path)
                assert result == tmp_path / "test_cs.sacc"

    def test_generate_factory_copies_template(self, tmp_path: Path) -> None:
        """Test that generate_factory copies template file."""
        with patch.object(ExampleCosmicShear, "_proceed_generation"):
            builder = ExampleCosmicShear(
                output_path=tmp_path,
                prefix="test_cs",
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        with patch("firecrown.app.examples._cosmic_shear.copy_template") as mock_copy:
            result = builder.generate_factory(tmp_path, sacc_path)

            mock_copy.assert_called_once()
            assert result == tmp_path / "test_cs_factory.py"

    def test_get_build_parameters(self, tmp_path: Path) -> None:
        """Test that get_build_parameters returns NamedParameters."""
        with patch.object(ExampleCosmicShear, "_proceed_generation"):
            builder = ExampleCosmicShear(
                output_path=tmp_path,
                prefix="test_cs",
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        params = builder.get_build_parameters(sacc_path)

        assert isinstance(params, NamedParameters)
        assert "sacc_file" in params.convert_to_basic_dict()

    def test_get_models_returns_delta_z_params(self, tmp_path: Path) -> None:
        """Test that get_models returns delta_z parameters for photo-z shifts."""
        with patch.object(ExampleCosmicShear, "_proceed_generation"):
            builder = ExampleCosmicShear(
                output_path=tmp_path,
                prefix="test_cs",
                n_bins=2,
                target_framework=Frameworks.COSMOSIS,
            )

        models = builder.get_models()

        assert isinstance(models, list)
        assert len(models) == 1
        # Should have delta_z parameters for each bin
        param_names = {p.name for p in models[0].parameters}
        assert any("delta_z" in name for name in param_names)

    def test_required_cosmology(self, tmp_path: Path) -> None:
        """Test that required_cosmology returns NONLINEAR."""
        with patch.object(ExampleCosmicShear, "_proceed_generation"):
            builder = ExampleCosmicShear(
                output_path=tmp_path,
                prefix="test_cs",
                target_framework=Frameworks.COSMOSIS,
            )

        result = builder.required_cosmology()
        assert result == FrameworkCosmology.NONLINEAR

    def test_get_options_desc(self, tmp_path: Path) -> None:
        """Test that get_options_desc returns configuration info."""
        with patch.object(ExampleCosmicShear, "_proceed_generation"):
            builder = ExampleCosmicShear(
                output_path=tmp_path,
                prefix="test_cs",
                n_bins=3,
                seed=100,
                target_framework=Frameworks.COSMOSIS,
            )

        options = builder.get_options_desc()

        assert isinstance(options, list)
        assert len(options) > 0
        # Should have configuration info
        option_names = [name for name, _ in options]
        assert any("bin" in name.lower() for name in option_names)
        assert any(
            "noise" in name.lower() or "ell" in name.lower() for name in option_names
        )
