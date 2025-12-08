"""Unit tests for firecrown.app.examples._sn_srd module.

Tests ExampleSupernovaSRD example generator without executing likelihoods.
"""

from pathlib import Path
from unittest.mock import patch

from firecrown.likelihood import NamedParameters
from firecrown.app.examples._sn_srd import ExampleSupernovaSRD
from firecrown.app.analysis import (
    Model,
    Parameter,
    FrameworkCosmology,
    CCLCosmologySpec,
    Frameworks,
)


class TestExampleSupernovaSRD:
    """Tests for ExampleSupernovaSRD class."""

    def test_class_attributes(self) -> None:
        """Test that class has required attributes."""
        assert hasattr(ExampleSupernovaSRD, "description")
        assert hasattr(ExampleSupernovaSRD, "data_url")
        assert "Supernova" in ExampleSupernovaSRD.description
        assert "github.com" in ExampleSupernovaSRD.data_url

    def test_generate_sacc_downloads_file(self, tmp_path: Path) -> None:
        """Test that generate_sacc calls download_from_url."""
        with patch.object(ExampleSupernovaSRD, "_proceed_generation"):
            builder = ExampleSupernovaSRD(
                output_path=tmp_path,
                prefix="test_sn",
                target_framework=Frameworks.COSMOSIS,
            )

        with patch("firecrown.app.examples._sn_srd.download_from_url") as mock_download:
            result = builder.generate_sacc(tmp_path)

            mock_download.assert_called_once()
            assert result == tmp_path / "test_sn.sacc"
            # Check that the URL was passed
            call_args = mock_download.call_args[0]
            assert ExampleSupernovaSRD.data_url == call_args[0]

    def test_generate_factory_copies_template(self, tmp_path: Path) -> None:
        """Test that generate_factory copies template file."""
        with patch.object(ExampleSupernovaSRD, "_proceed_generation"):
            builder = ExampleSupernovaSRD(
                output_path=tmp_path,
                prefix="test_sn",
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        with patch("firecrown.app.examples._sn_srd.copy_template") as mock_copy:
            result = builder.generate_factory(tmp_path, sacc_path)

            mock_copy.assert_called_once()
            assert result == tmp_path / "test_sn_factory.py"

    def test_get_build_parameters(self, tmp_path: Path) -> None:
        """Test that get_build_parameters returns NamedParameters with sacc_file."""
        with patch.object(ExampleSupernovaSRD, "_proceed_generation"):
            builder = ExampleSupernovaSRD(
                output_path=tmp_path,
                prefix="test_sn",
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test.sacc"
        params = builder.get_build_parameters(sacc_path)

        assert isinstance(params, NamedParameters)
        assert "sacc_file" in params.convert_to_basic_dict()

    def test_get_models(self, tmp_path: Path) -> None:
        """Test that get_models returns list with SN magnitude parameter."""
        with patch.object(ExampleSupernovaSRD, "_proceed_generation"):
            builder = ExampleSupernovaSRD(
                output_path=tmp_path,
                prefix="test_sn",
                target_framework=Frameworks.COSMOSIS,
            )

        models = builder.get_models()

        assert isinstance(models, list)
        assert len(models) == 1
        assert isinstance(models[0], Model)
        assert models[0].name == "firecrown_test_sn"
        assert len(models[0].parameters) == 1

        param = models[0].parameters[0]
        assert isinstance(param, Parameter)
        assert param.name == "sn_ddf_sample_M"
        assert param.free is True
        assert param.prior is not None

    def test_required_cosmology(self, tmp_path: Path) -> None:
        """Test that required_cosmology returns BACKGROUND."""
        with patch.object(ExampleSupernovaSRD, "_proceed_generation"):
            builder = ExampleSupernovaSRD(
                output_path=tmp_path,
                prefix="test_sn",
                target_framework=Frameworks.COSMOSIS,
            )

        result = builder.required_cosmology()
        assert result == FrameworkCosmology.BACKGROUND

    def test_cosmology_analysis_spec(self, tmp_path: Path) -> None:
        """Test that cosmology_analysis_spec returns CCLCosmologySpec with A_s."""
        with patch.object(ExampleSupernovaSRD, "_proceed_generation"):
            builder = ExampleSupernovaSRD(
                output_path=tmp_path,
                prefix="test_sn",
                target_framework=Frameworks.COSMOSIS,
            )

        spec = builder.cosmology_analysis_spec()

        assert isinstance(spec, CCLCosmologySpec)
        param_names = {p.name for p in spec.parameters}
        # Should have A_s instead of sigma8
        assert "A_s" in param_names
        # Should have minimal required parameters
        assert "Omega_c" in param_names
        assert "Omega_b" in param_names
        assert "h" in param_names

    def test_default_prefix(self, tmp_path: Path) -> None:
        """Test that default prefix is 'sn_srd'."""
        with patch.object(ExampleSupernovaSRD, "_proceed_generation"):
            builder = ExampleSupernovaSRD(
                output_path=tmp_path,
                target_framework=Frameworks.COSMOSIS,
            )

        assert builder.prefix == "sn_srd"
