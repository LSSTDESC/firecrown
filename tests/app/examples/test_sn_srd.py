"""Unit tests for firecrown.app.examples._sn_srd module.

Tests ExampleSupernovaSRD example generator and build_likelihood execution.
"""

import sys
from pathlib import Path
from unittest.mock import patch
import importlib.util

from firecrown.likelihood import NamedParameters, ConstGaussian
from firecrown.modeling_tools import ModelingTools
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

    def test_build_likelihood_execution(self, tmp_path: Path) -> None:
        """Test that build_likelihood from template executes successfully."""
        # Generate real SACC file and factory (downloads from GitHub)
        builder = ExampleSupernovaSRD(
            output_path=tmp_path,
            prefix="test_sn",
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        factory_file = builder.generate_factory(tmp_path, sacc_file)
        params = NamedParameters({"sacc_file": str(sacc_file)})

        # Test 1: Load and execute the ORIGINAL template module for coverage
        from firecrown.app.examples import _sn_srd_template

        likelihood_orig, modeling_tools_orig = _sn_srd_template.build_likelihood(params)
        assert isinstance(likelihood_orig, ConstGaussian)
        assert isinstance(modeling_tools_orig, ModelingTools)
        assert len(likelihood_orig.statistics) == 1  # Single supernova statistic

        # Test 2: Load and execute the COPIED template module
        spec = importlib.util.spec_from_file_location("test_sn_factory", factory_file)
        assert spec is not None
        assert spec.loader is not None
        factory_module = importlib.util.module_from_spec(spec)
        sys.modules["test_sn_factory"] = factory_module
        spec.loader.exec_module(factory_module)

        likelihood_copy, modeling_tools_copy = factory_module.build_likelihood(params)
        assert isinstance(likelihood_copy, ConstGaussian)
        assert isinstance(modeling_tools_copy, ModelingTools)
        assert len(likelihood_copy.statistics) == 1

        # Verify both produce equivalent results
        assert len(likelihood_orig.statistics) == len(likelihood_copy.statistics)

        # Clean up
        del sys.modules["test_sn_factory"]
