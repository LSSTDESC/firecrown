"""Unit tests for firecrown.app.examples._des_y1_3x2pt module.

Tests ExampleDESY13x2pt example generator and build_likelihood execution.
"""

import sys
from pathlib import Path
from unittest.mock import patch
import importlib.util

import numpy as np
import pyccl

from firecrown.likelihood import NamedParameters, ConstGaussian
from firecrown.modeling_tools import ModelingTools, PowerspectrumModifier
from firecrown.app.examples._des_y1_3x2pt import (
    ExampleDESY13x2pt,
    DESY1FactoryType,
    _des_y1_3x2pt_template,
    _des_y1_3x2pt_pt_template,
    _des_y1_cosmic_shear_hmia_template,
    _des_y1_cosmic_shear_pk_modifier_template,
    _des_y1_cosmic_shear_tatt_template,
)
from firecrown.app.analysis import (
    Model,
    FrameworkCosmology,
    Frameworks,
)
from firecrown.likelihood.factories import build_two_point_likelihood
from firecrown.updatable import ParamsMap


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

    def test_build_likelihood_execution_standard(self, tmp_path: Path) -> None:
        """Test that build_likelihood from standard template executes."""
        # Generate real SACC file and factory (downloads from GitHub)
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des",
            factory_type=DESY1FactoryType.STANDARD,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        factory_file = builder.generate_factory(tmp_path, sacc_file)
        params = NamedParameters({"sacc_file": str(sacc_file)})

        # Test 1: Load and execute the ORIGINAL template module for coverage

        likelihood_orig, modeling_tools_orig = _des_y1_3x2pt_template.build_likelihood(
            params
        )
        assert isinstance(likelihood_orig, ConstGaussian)
        assert isinstance(modeling_tools_orig, ModelingTools)
        # DES Y1 has cosmic shear (xip/xim), galaxy-galaxy lensing, clustering
        assert len(likelihood_orig.statistics) > 0
        _ = len(likelihood_orig.statistics)

        # Test 2: Load and execute the COPIED template module
        spec = importlib.util.spec_from_file_location("test_des_factory", factory_file)
        assert spec is not None
        assert spec.loader is not None
        factory_module = importlib.util.module_from_spec(spec)
        sys.modules["test_des_factory"] = factory_module
        spec.loader.exec_module(factory_module)

        likelihood_copy, modeling_tools_copy = factory_module.build_likelihood(params)
        assert isinstance(likelihood_copy, ConstGaussian)
        assert isinstance(modeling_tools_copy, ModelingTools)
        assert len(likelihood_copy.statistics) > 0

        # Verify both produce equivalent results
        assert len(likelihood_orig.statistics) == len(likelihood_copy.statistics)

        # Clean up
        del sys.modules["test_des_factory"]

    def test_build_likelihood_execution_pt(self, tmp_path: Path) -> None:
        """Test that build_likelihood from PT template executes."""
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des_pt",
            factory_type=DESY1FactoryType.PT,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        factory_file = builder.generate_factory(tmp_path, sacc_file)
        params = NamedParameters({"sacc_file": str(sacc_file)})

        # Test 1: Load and execute the ORIGINAL template module for coverage
        likelihood_orig, modeling_tools_orig = (
            _des_y1_3x2pt_pt_template.build_likelihood(params)
        )
        assert isinstance(likelihood_orig, ConstGaussian)
        assert isinstance(modeling_tools_orig, ModelingTools)
        assert len(likelihood_orig.statistics) > 0

        # Test 2: Load and execute the COPIED template module
        spec = importlib.util.spec_from_file_location(
            "test_des_pt_factory", factory_file
        )
        assert spec is not None
        assert spec.loader is not None
        factory_module = importlib.util.module_from_spec(spec)
        sys.modules["test_des_pt_factory"] = factory_module
        spec.loader.exec_module(factory_module)

        likelihood_copy, modeling_tools_copy = factory_module.build_likelihood(params)
        assert isinstance(likelihood_copy, ConstGaussian)
        assert isinstance(modeling_tools_copy, ModelingTools)
        assert len(likelihood_copy.statistics) > 0

        # Verify both produce equivalent results
        assert len(likelihood_orig.statistics) == len(likelihood_copy.statistics)

        # Clean up
        del sys.modules["test_des_pt_factory"]

    def test_build_likelihood_tatt_template(self, tmp_path: Path) -> None:
        """Test that TATT template build_likelihood executes."""
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des_tatt",
            factory_type=DESY1FactoryType.TATT,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        params = NamedParameters({"sacc_file": str(sacc_file)})

        # Load and execute the ORIGINAL TATT template module for coverage
        likelihood, modeling_tools = (
            _des_y1_cosmic_shear_tatt_template.build_likelihood(params)
        )
        assert isinstance(likelihood, ConstGaussian)
        assert isinstance(modeling_tools, ModelingTools)
        assert len(likelihood.statistics) > 0

    def test_build_likelihood_hmia_template(self, tmp_path: Path) -> None:
        """Test that HMIA template build_likelihood executes."""
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des_hmia",
            factory_type=DESY1FactoryType.HMIA,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        params = NamedParameters({"sacc_file": str(sacc_file)})

        # Load and execute the ORIGINAL HMIA template module for coverage
        likelihood, modeling_tools = (
            _des_y1_cosmic_shear_hmia_template.build_likelihood(params)
        )
        assert isinstance(likelihood, ConstGaussian)
        assert isinstance(modeling_tools, ModelingTools)
        assert len(likelihood.statistics) > 0

    def test_build_likelihood_pk_modifier_template(self, tmp_path: Path) -> None:
        """Test that PK_MODIFIER template build_likelihood executes."""
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des_pk",
            factory_type=DESY1FactoryType.PK_MODIFIER,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        params = NamedParameters({"sacc_file": str(sacc_file)})

        # Load and execute the ORIGINAL PK_MODIFIER template module for coverage
        likelihood, modeling_tools = (
            _des_y1_cosmic_shear_pk_modifier_template.build_likelihood(params)
        )
        assert isinstance(likelihood, ConstGaussian)
        assert isinstance(modeling_tools, ModelingTools)
        assert len(likelihood.statistics) > 0

    def test_pk_modifier_compute_p_of_k_z(self, tmp_path: Path) -> None:
        """Test that PK_MODIFIER's compute_p_of_k_z method executes correctly."""
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des_pk",
            factory_type=DESY1FactoryType.PK_MODIFIER,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        params = NamedParameters({"sacc_file": str(sacc_file)})

        # Build likelihood and get modeling tools with PK modifier
        likelihood, modeling_tools = (
            _des_y1_cosmic_shear_pk_modifier_template.build_likelihood(params)
        )

        # Verify that pk_modifiers are present
        assert len(modeling_tools.pk_modifiers) == 1
        # Hack to make pylint happy about the type of pk_modifier
        pk_modifier: PowerspectrumModifier = next(iter(modeling_tools.pk_modifiers))
        assert pk_modifier is not None
        assert isinstance(pk_modifier, PowerspectrumModifier)

        # Check that it's the right type
        assert pk_modifier.__class__.__name__ == "vanDaalen19Baryonfication"
        assert hasattr(pk_modifier, "compute_p_of_k_z")
        assert hasattr(pk_modifier, "f_bar")

        # Prepare modeling tools with a complete set of cosmological parameters
        # CCLFactory requires: Omega_c, Omega_b, h, n_s, Omega_k, Neff, m_nu,
        # w0, wa, T_CMB, and sigma8 (amplitude parameter)
        test_params = ParamsMap(
            {
                "Omega_c": 0.27,
                "Omega_b": 0.045,
                "h": 0.67,
                "n_s": 0.96,
                "Omega_k": 0.0,
                "Neff": 3.046,
                "m_nu": 0.06,
                "w0": -1.0,
                "wa": 0.0,
                "T_CMB": 2.7255,
                "sigma8": 0.8,
                "f_bar": 0.5,
                "src0_delta_z": 0.0,
            }
        )
        likelihood.update(test_params)
        modeling_tools.update(test_params)
        modeling_tools.prepare()

        # Exercise compute_p_of_k_z method
        pk_modified = pk_modifier.compute_p_of_k_z(modeling_tools)

        # Verify the result is a valid Pk2D object
        assert isinstance(pk_modified, pyccl.Pk2D)

        # Test that we can evaluate the power spectrum at some k and z
        k_test = np.array([0.1, 1.0, 10.0])
        z_test = 0.5
        pk_values = pk_modified(k_test, 1.0 / (1.0 + z_test))

        # Check that we get valid power spectrum values
        assert isinstance(pk_values, np.ndarray)
        assert len(pk_values) == len(k_test)
        assert np.all(np.isfinite(pk_values))
        assert np.all(pk_values > 0)  # Power spectrum should be positive

        # Verify that changing f_bar affects the result
        test_params_2 = ParamsMap(
            {
                "Omega_c": 0.27,
                "Omega_b": 0.045,
                "h": 0.67,
                "n_s": 0.96,
                "Omega_k": 0.0,
                "Neff": 3.046,
                "m_nu": 0.06,
                "w0": -1.0,
                "wa": 0.0,
                "T_CMB": 2.7255,
                "sigma8": 0.8,
                "f_bar": 0.8,  # Different f_bar value
                "src0_delta_z": 0.0,
            }
        )
        likelihood.reset()
        modeling_tools.reset()
        likelihood.update(test_params_2)
        modeling_tools.update(test_params_2)
        modeling_tools.prepare()

        pk_modified_2 = pk_modifier.compute_p_of_k_z(modeling_tools)
        pk_values_2 = pk_modified_2(k_test, 1.0 / (1.0 + z_test))

        # Values should be different when f_bar changes
        assert not np.allclose(pk_values, pk_values_2)

    def test_build_likelihood_yaml_default(self, tmp_path: Path) -> None:
        """Test that YAML_DEFAULT factory build_likelihood executes."""
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des_yaml",
            factory_type=DESY1FactoryType.YAML_DEFAULT,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        factory_result = builder.generate_factory(tmp_path, sacc_file)

        # YAML factories return a string with the factory function path
        assert isinstance(factory_result, str)
        assert "build_two_point_likelihood" in factory_result

        # Verify YAML file was created
        yaml_file = tmp_path / "test_des_yaml_experiment.yaml"
        assert yaml_file.exists()
        yaml_content = yaml_file.read_text()
        assert "sacc_data_file" in yaml_content
        assert "two_point_factory" in yaml_content

        # Execute build_likelihood using the YAML factory
        params = builder.get_build_parameters(sacc_file)
        assert "likelihood_config" in params.convert_to_basic_dict()

        likelihood, modeling_tools = build_two_point_likelihood(params)

        assert isinstance(likelihood, ConstGaussian)
        assert isinstance(modeling_tools, ModelingTools)
        assert len(likelihood.statistics) > 0

    def test_build_likelihood_yaml_pure_ccl(self, tmp_path: Path) -> None:
        """Test that YAML_PURE_CCL factory build_likelihood executes."""
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des_pure_ccl",
            factory_type=DESY1FactoryType.YAML_PURE_CCL,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        factory_result = builder.generate_factory(tmp_path, sacc_file)

        assert isinstance(factory_result, str)
        assert "build_two_point_likelihood" in factory_result

        # Verify YAML file contains pure_ccl_mode
        yaml_file = tmp_path / "test_des_pure_ccl_experiment.yaml"
        assert yaml_file.exists()
        yaml_content = yaml_file.read_text()
        assert "pure_ccl_mode" in yaml_content
        assert "ccl_factory" in yaml_content

        # Execute build_likelihood using the YAML factory
        params = builder.get_build_parameters(sacc_file)
        likelihood, modeling_tools = build_two_point_likelihood(params)

        assert isinstance(likelihood, ConstGaussian)
        assert isinstance(modeling_tools, ModelingTools)
        assert len(likelihood.statistics) > 0

    def test_build_likelihood_yaml_mu_sigma(self, tmp_path: Path) -> None:
        """Test that YAML_MU_SIGMA factory build_likelihood executes."""
        builder = ExampleDESY13x2pt(
            output_path=tmp_path,
            prefix="test_des_mu_sigma",
            factory_type=DESY1FactoryType.YAML_MU_SIGMA,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        factory_result = builder.generate_factory(tmp_path, sacc_file)

        assert isinstance(factory_result, str)
        assert "build_two_point_likelihood" in factory_result

        # Verify YAML file contains mu_sigma_isitgr
        yaml_file = tmp_path / "test_des_mu_sigma_experiment.yaml"
        assert yaml_file.exists()
        yaml_content = yaml_file.read_text()
        assert "mu_sigma_isitgr" in yaml_content
        assert "ccl_factory" in yaml_content

        # Execute build_likelihood using the YAML factory
        params = builder.get_build_parameters(sacc_file)

        likelihood, modeling_tools = build_two_point_likelihood(params)

        assert isinstance(likelihood, ConstGaussian)
        assert isinstance(modeling_tools, ModelingTools)
        assert len(likelihood.statistics) > 0
