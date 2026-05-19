"""Unit tests for firecrown.app.analysis._analysis_builder module.

Tests AnalysisBuilder base class and workflow orchestration.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._analysis_builder import AnalysisBuilder
from firecrown.app.analysis._types import (
    Frameworks,
    FrameworkCosmology,
    CCLCosmologySpec,
    Model,
    Parameter,
)
from firecrown.app.sacc import SaccFormat


class ConcreteAnalysisBuilder(AnalysisBuilder):
    """Concrete implementation of AnalysisBuilder for testing."""

    description = "Test analysis for unit tests"

    def generate_sacc(self, output_path: Path) -> Path:
        """Generate mock SACC file."""
        sacc_file = output_path / f"{self.prefix}.sacc"
        sacc_file.touch()
        return sacc_file

    def generate_factory(self, output_path: Path, sacc: Path) -> str | Path:
        """Generate mock factory file."""
        factory_file = output_path / f"{self.prefix}_factory.py"
        factory_file.write_text("def build_likelihood(params):\n    return None\n")
        return factory_file

    def get_build_parameters(self, sacc_path: Path) -> NamedParameters:
        """Return mock build parameters."""
        return NamedParameters({"sacc_file": str(sacc_path)})

    def get_models(self) -> list[Model]:
        """Return empty model list."""
        return []

    def required_cosmology(self) -> FrameworkCosmology:
        """Require nonlinear cosmology."""
        return FrameworkCosmology.NONLINEAR


@pytest.fixture(name="vanilla_cosmo")
def fixture_vanilla_cosmo() -> CCLCosmologySpec:
    """Create vanilla LCDM cosmology spec for testing."""
    return CCLCosmologySpec.vanilla_lcdm()


class TestAnalysisBuilderInitialization:
    """Tests for AnalysisBuilder initialization."""

    def test_initialization_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates output directory."""
        output_dir = tmp_path / "nonexistent"
        assert not output_dir.exists()

        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            _ = ConcreteAnalysisBuilder(
                output_path=output_dir,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_initialization_with_existing_directory(self, tmp_path: Path) -> None:
        """Test initialization with existing output directory."""
        tmp_path.mkdir(exist_ok=True)

        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            _ = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            assert tmp_path.exists()

    def test_initialization_with_custom_cosmology(self, tmp_path: Path) -> None:
        """Test initialization with custom cosmology specification."""
        # Create a cosmology spec file
        cosmo_file = tmp_path / "cosmology.yaml"
        cosmo_file.write_text(r"""
name: test_cosmology
description: Test cosmology
parameters:
  - name: Omega_c
    symbol: "\\Omega_c"
    lower_bound: 0.2
    upper_bound: 0.3
    default_value: 0.25
    free: true
  - name: Omega_b
    symbol: "\\Omega_b"
    lower_bound: 0.03
    upper_bound: 0.07
    default_value: 0.05
    free: true
  - name: h
    symbol: h
    lower_bound: 0.6
    upper_bound: 0.8
    default_value: 0.67
    free: false
  - name: n_s
    symbol: n_s
    lower_bound: 0.87
    upper_bound: 1.07
    default_value: 0.96
    free: false
  - name: sigma8
    symbol: "\\sigma_8"
    lower_bound: 0.6
    upper_bound: 1.0
    default_value: 0.81
    free: true
  - name: Omega_k
    symbol: "\\Omega_k"
    lower_bound: -0.2
    upper_bound: 0.2
    default_value: 0.0
    free: false
  - name: Neff
    symbol: N_eff
    lower_bound: 2.0
    upper_bound: 5.0
    default_value: 3.046
    free: false
  - name: m_nu
    symbol: m_nu
    lower_bound: 0.0
    upper_bound: 5.0
    default_value: 0.0
    free: false
  - name: w0
    symbol: w_0
    lower_bound: -3.0
    upper_bound: 0.0
    default_value: -1.0
    free: false
  - name: wa
    symbol: w_a
    lower_bound: -1.0
    upper_bound: 1.0
    default_value: 0.0
    free: false
""")

        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path / "output",
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
                cosmology_spec=cosmo_file,
            )
            # pylint: disable=protected-access
            assert builder._spec is not None
            assert builder._spec.name == "test_cosmology"
            # pylint: enable=protected-access

    def test_initialization_with_nonexistent_cosmology_raises(
        self, tmp_path: Path
    ) -> None:
        """Test that nonexistent cosmology file raises error."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(ValueError, match="does not exist"):
            with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
                ConcreteAnalysisBuilder(
                    output_path=tmp_path / "output",
                    prefix="test",
                    target_framework=Frameworks.COSMOSIS,
                    cosmology_spec=nonexistent,
                )


class TestAnalysisBuilderMethods:
    """Tests for AnalysisBuilder methods."""

    def test_get_sacc_file_absolute_path(self, tmp_path: Path) -> None:
        """Test get_sacc_file with absolute path."""
        sacc_path = tmp_path / "data.sacc"

        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
                use_absolute_path=True,
            )

            result = builder.get_sacc_file(sacc_path)
            assert str(sacc_path.absolute()) in result

    def test_get_sacc_file_relative_path(self, tmp_path: Path) -> None:
        """Test get_sacc_file with relative path."""
        sacc_path = tmp_path / "data.sacc"

        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
                use_absolute_path=False,
            )

            result = builder.get_sacc_file(sacc_path)
            assert result == "data.sacc"

    def test_cosmology_analysis_spec_default(self, tmp_path: Path) -> None:
        """Test cosmology_analysis_spec returns vanilla LCDM by default."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            spec = builder.cosmology_analysis_spec()
            assert spec.name == "ccl_cosmology"
            assert "Omega_c" in spec

    def test_get_options_desc_default(self, tmp_path: Path) -> None:
        """Test get_options_desc returns empty by default."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            result = builder.get_options_desc()
            assert not result


class TestAnalysisBuilderDisplay:
    """Tests for AnalysisBuilder display methods."""

    def test_display_with_options_desc(self, tmp_path: Path, capsys) -> None:
        """Test that display shows options when get_options_desc returns values."""

        class BuilderWithOptions(ConcreteAnalysisBuilder):
            """Builder that returns options."""

            def get_options_desc(self) -> list[tuple[str, str]]:
                """Return sample options."""
                return [("option1", "description1"), ("option2", "description2")]

        with patch.object(BuilderWithOptions, "_proceed_generation"):
            _ = BuilderWithOptions(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )
            # The display already happened in __post_init__, captured by capsys
            captured = capsys.readouterr()
            assert "option1" in captured.out
            assert "description1" in captured.out
            assert "option2" in captured.out
            assert "description2" in captured.out


class TestAnalysisBuilderAttributes:
    """Tests for AnalysisBuilder attributes."""

    def test_description_attribute(self) -> None:
        """Test that ConcreteAnalysisBuilder has description."""
        assert hasattr(ConcreteAnalysisBuilder, "description")
        assert ConcreteAnalysisBuilder.description == "Test analysis for unit tests"

    def test_required_methods_exist(self) -> None:
        """Test that all required abstract methods are implemented."""
        required_methods = [
            "generate_sacc",
            "generate_factory",
            "get_build_parameters",
            "get_models",
            "required_cosmology",
        ]

        for method in required_methods:
            assert hasattr(ConcreteAnalysisBuilder, method)


class TestAnalysisBuilderWorkflow:
    """Tests for AnalysisBuilder workflow."""

    def test_workflow_calls_generate_sacc(self, tmp_path: Path) -> None:
        """Test that workflow calls generate_sacc."""
        # Create actual SACC file
        sacc_file = tmp_path / "data.sacc"
        sacc_file.touch()

        with patch.object(
            ConcreteAnalysisBuilder,
            "generate_sacc",
            return_value=sacc_file,
        ) as mock_gen_sacc:
            with patch.object(ConcreteAnalysisBuilder, "generate_factory"):
                with patch.object(ConcreteAnalysisBuilder, "get_build_parameters"):
                    with patch.object(ConcreteAnalysisBuilder, "get_models"):
                        with patch(
                            "firecrown.app.analysis._analysis_builder.get_generator"
                        ):
                            with patch(
                                "firecrown.app.analysis._analysis_builder.Transform"
                            ) as mock_transform:
                                # Mock the detect_format static method
                                mock_transform.detect_format.return_value = (
                                    SaccFormat.FITS
                                )
                                _ = ConcreteAnalysisBuilder(
                                    output_path=tmp_path,
                                    prefix="test",
                                    target_framework=Frameworks.COSMOSIS,
                                )

                                mock_gen_sacc.assert_called_once()

    def test_workflow_calls_generate_factory(self, tmp_path: Path) -> None:
        """Test that workflow calls generate_factory."""
        # Create actual SACC file
        sacc_file = tmp_path / "data.sacc"
        sacc_file.touch()

        with patch.object(
            ConcreteAnalysisBuilder, "generate_factory"
        ) as mock_gen_factory:
            mock_gen_factory.return_value = tmp_path / "factory.py"

            with patch.object(
                ConcreteAnalysisBuilder,
                "generate_sacc",
                return_value=sacc_file,
            ):
                with patch.object(ConcreteAnalysisBuilder, "get_build_parameters"):
                    with patch.object(ConcreteAnalysisBuilder, "get_models"):
                        with patch(
                            "firecrown.app.analysis._analysis_builder.get_generator"
                        ):
                            with patch(
                                "firecrown.app.analysis._analysis_builder.Transform"
                            ) as mock_transform:
                                # Mock the detect_format static method
                                mock_transform.detect_format.return_value = (
                                    SaccFormat.FITS
                                )
                                _ = ConcreteAnalysisBuilder(
                                    output_path=tmp_path,
                                    prefix="test",
                                    target_framework=Frameworks.COSMOSIS,
                                )

                                mock_gen_factory.assert_called_once()


class TestSaccFormatHandling:
    """Tests for SACC format handling."""

    def test_default_sacc_format(self, tmp_path: Path) -> None:
        """Test default SACC format is HDF5."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            assert builder.sacc_format == SaccFormat.HDF5

    def test_custom_sacc_format(self, tmp_path: Path) -> None:
        """Test setting custom SACC format."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
                sacc_format=SaccFormat.FITS,
            )

            assert builder.sacc_format == SaccFormat.FITS

    def test_sacc_no_conversion_needed(self, tmp_path: Path) -> None:
        """Test that no conversion happens when SACC is already in target format."""
        # Create actual SACC file
        sacc_file = tmp_path / "data.sacc"
        sacc_file.touch()

        with patch.object(
            ConcreteAnalysisBuilder,
            "generate_sacc",
            return_value=sacc_file,
        ):
            with patch.object(
                ConcreteAnalysisBuilder,
                "generate_factory",
                return_value=tmp_path / "factory.py",
            ):
                with patch.object(
                    ConcreteAnalysisBuilder,
                    "get_build_parameters",
                    return_value=NamedParameters({}),
                ):
                    with patch.object(
                        ConcreteAnalysisBuilder, "get_models", return_value=[]
                    ):
                        with patch(
                            "firecrown.app.analysis._analysis_builder.get_generator"
                        ) as mock_get_gen:
                            mock_generator = mock_get_gen.return_value
                            mock_generator.add_sacc.return_value = None
                            mock_generator.add_factory.return_value = None
                            mock_generator.add_build_parameters.return_value = None
                            mock_generator.add_models.return_value = None
                            mock_generator.write_config.return_value = None

                            with patch(
                                "firecrown.app.analysis._analysis_builder.Transform"
                            ) as mock_transform:
                                # Mock detect_format to return HDF5 (same as default)
                                mock_transform.detect_format.return_value = (
                                    SaccFormat.HDF5
                                )

                                _ = ConcreteAnalysisBuilder(
                                    output_path=tmp_path,
                                    prefix="test",
                                    target_framework=Frameworks.COSMOSIS,
                                    sacc_format=SaccFormat.HDF5,
                                )

                                # Transform should only have detect_format called, not
                                # instantiated
                                mock_transform.detect_format.assert_called_once_with(
                                    sacc_file
                                )
                                # Transform itself should not be instantiated
                                assert mock_transform.call_count == 0
                                # Verify the generator received the original file (not
                                # converted)
                                mock_generator.add_sacc.assert_called_once_with(
                                    sacc_file
                                )


class TestFrameworkSelection:
    """Tests for framework selection."""

    def test_cosmosis_framework(self, tmp_path: Path) -> None:
        """Test selecting CosmoSIS framework."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            assert builder.target_framework == Frameworks.COSMOSIS

    def test_cobaya_framework(self, tmp_path: Path) -> None:
        """Test selecting Cobaya framework."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COBAYA,
            )

            assert builder.target_framework == Frameworks.COBAYA

    def test_numcosmo_framework(self, tmp_path: Path) -> None:
        """Test selecting NumCosmo framework."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.NUMCOSMO,
            )

            assert builder.target_framework == Frameworks.NUMCOSMO


class TestPrefixHandling:
    """Tests for prefix handling."""

    def test_prefix_storage(self, tmp_path: Path) -> None:
        """Test that prefix is stored correctly."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="my_analysis",
                target_framework=Frameworks.COSMOSIS,
            )

            assert builder.prefix == "my_analysis"

    def test_different_prefixes(self, tmp_path: Path) -> None:
        """Test that different prefixes work."""
        prefixes = ["test", "analysis1", "des_y1_3x2pt", "cosmic_shear"]

        for prefix in prefixes:
            with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
                builder = ConcreteAnalysisBuilder(
                    output_path=tmp_path / prefix,
                    prefix=prefix,
                    target_framework=Frameworks.COSMOSIS,
                )

                assert builder.prefix == prefix


class TestAbstractMethodRequirements:
    """Tests that abstract methods must be implemented."""

    def test_cannot_instantiate_base_class(self, tmp_path: Path) -> None:
        """Test that AnalysisBuilder cannot be instantiated directly."""
        # AnalysisBuilder is abstract but Python dataclasses allow instantiation
        # It will fail when trying to access the description class variable
        with pytest.raises((TypeError, AttributeError)):
            AnalysisBuilder(  # type: ignore
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

    def test_concrete_class_implements_all_methods(self) -> None:
        """Test that ConcreteAnalysisBuilder implements all required methods."""
        abstract_methods = [
            "generate_sacc",
            "generate_factory",
            "get_build_parameters",
            "get_models",
            "required_cosmology",
        ]

        for method_name in abstract_methods:
            assert hasattr(ConcreteAnalysisBuilder, method_name)
            method = getattr(ConcreteAnalysisBuilder, method_name)
            assert callable(method)


class TestModelHandling:
    """Tests for model parameter handling."""

    def test_get_models_returns_list(self, tmp_path: Path) -> None:
        """Test that get_models returns a list."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            models = builder.get_models()
            assert isinstance(models, list)

    def test_builder_with_models(self, tmp_path: Path) -> None:
        """Test builder with custom models."""

        class BuilderWithModels(ConcreteAnalysisBuilder):
            """Builder that returns sample models."""

            def get_models(self) -> list[Model]:
                return [
                    Model(
                        name="test_model",
                        description="Test",
                        parameters=[
                            Parameter(
                                name="param1",
                                symbol="p1",
                                lower_bound=0.0,
                                upper_bound=1.0,
                                default_value=0.5,
                                free=True,
                            )
                        ],
                    )
                ]

        with patch.object(BuilderWithModels, "_proceed_generation"):
            builder = BuilderWithModels(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            models = builder.get_models()
            assert len(models) == 1
            assert models[0].name == "test_model"


class TestBuildParametersHandling:
    """Tests for build parameters handling."""

    def test_get_build_parameters_returns_named_parameters(
        self, tmp_path: Path
    ) -> None:
        """Test that get_build_parameters returns NamedParameters."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            sacc_path = tmp_path / "data.sacc"
            params = builder.get_build_parameters(sacc_path)
            assert isinstance(params, NamedParameters)

    def test_build_parameters_include_sacc_file(self, tmp_path: Path) -> None:
        """Test that build parameters include sacc_file."""
        with patch.object(ConcreteAnalysisBuilder, "_proceed_generation"):
            builder = ConcreteAnalysisBuilder(
                output_path=tmp_path,
                prefix="test",
                target_framework=Frameworks.COSMOSIS,
            )

            sacc_path = tmp_path / "data.sacc"
            params = builder.get_build_parameters(sacc_path)
            params_dict = params.convert_to_basic_dict()
            assert "sacc_file" in params_dict
