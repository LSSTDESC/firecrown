"""Unit tests for firecrown.app.analysis config generator module.

Tests framework configuration generation and utilities.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from firecrown.app.analysis import (
    Frameworks,
    FrameworkCosmology,
    get_generator,
)


@pytest.fixture(name="_mock_cosmo_spec")
def fixture__mock_cosmo_spec() -> MagicMock:
    """Create a mock cosmology spec for testing."""
    return MagicMock(name="mock_cosmology_spec")


class TestGetGenerator:
    """Tests for get_generator factory function."""

    def test_get_generator_requires_output_path(
        self, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that get_generator requires output_path parameter."""
        with_path = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=Path("/tmp/test"),
            prefix="test_analysis",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert with_path is not None

    def test_get_generator_accepts_all_frameworks(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that get_generator works with all framework types."""
        for framework in Frameworks:
            generator = get_generator(
                framework=framework,
                output_path=tmp_path,
                prefix="test_analysis",
                use_absolute_path=True,
                cosmo_spec=_mock_cosmo_spec,
                required_cosmology=FrameworkCosmology.NONLINEAR,
            )
            assert generator is not None
            assert generator.framework == framework

    def test_get_generator_prefix_stored(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that generator stores the provided prefix."""
        prefix = "my_test_analysis"
        generator = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix=prefix,
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert generator.prefix == prefix

    def test_get_generator_output_path_stored(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that generator stores the provided output_path."""
        generator = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert generator.output_path == tmp_path

    def test_get_generator_cosmology_type(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that generator stores cosmology type."""
        for cosmo_type in FrameworkCosmology:
            generator = get_generator(
                framework=Frameworks.COSMOSIS,
                output_path=tmp_path,
                prefix="test",
                use_absolute_path=True,
                cosmo_spec=_mock_cosmo_spec,
                required_cosmology=cosmo_type,
            )
            assert generator.required_cosmology == cosmo_type

    def test_get_generator_absolute_path_flag(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that generator respects use_absolute_path flag."""
        gen_abs = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.LINEAR,
        )
        gen_rel = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=False,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.LINEAR,
        )
        assert gen_abs.use_absolute_path is True
        assert gen_rel.use_absolute_path is False

    def test_get_generator_cobaya_framework(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test getting Cobaya generator specifically."""
        generator = get_generator(
            framework=Frameworks.COBAYA,
            output_path=tmp_path,
            prefix="cobaya_test",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert generator.framework == Frameworks.COBAYA

    def test_get_generator_numcosmo_framework(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test getting NumCosmo generator specifically."""
        generator = get_generator(
            framework=Frameworks.NUMCOSMO,
            output_path=tmp_path,
            prefix="numcosmo_test",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert generator.framework == Frameworks.NUMCOSMO

    def test_get_generator_different_prefixes(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that generators can be created with different prefixes."""
        gen1 = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix="analysis_one",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.LINEAR,
        )
        gen2 = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix="analysis_two",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.LINEAR,
        )
        assert gen1.prefix == "analysis_one"
        assert gen2.prefix == "analysis_two"
        assert gen1.prefix != gen2.prefix


class TestConfigGeneratorInterface:
    """Tests for ConfigGenerator base interface."""

    def test_generator_has_framework_attribute(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that generator has framework attribute."""
        generator = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert hasattr(generator, "framework")

    def test_generator_has_prefix_attribute(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that generator has prefix attribute."""
        generator = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix="test_prefix",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert hasattr(generator, "prefix")
        assert generator.prefix == "test_prefix"

    def test_generator_has_output_path_attribute(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test that generator has output_path attribute."""
        generator = get_generator(
            framework=Frameworks.COSMOSIS,
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=_mock_cosmo_spec,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert hasattr(generator, "output_path")
        assert generator.output_path == tmp_path

    def test_generator_cosmosis_type(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test CosmoSIS generator initialization."""
        generator = get_generator(
            Frameworks.COSMOSIS,
            tmp_path,
            "test_analysis",
            True,
            _mock_cosmo_spec,
            FrameworkCosmology.NONLINEAR,
        )
        assert generator.framework == Frameworks.COSMOSIS
        assert generator.prefix == "test_analysis"
        assert generator.output_path == tmp_path
        assert generator.use_absolute_path is True

    def test_generator_cobaya_type(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test Cobaya generator initialization."""
        generator = get_generator(
            Frameworks.COBAYA,
            tmp_path,
            "test_analysis",
            True,
            _mock_cosmo_spec,
            FrameworkCosmology.NONLINEAR,
        )
        assert generator.framework == Frameworks.COBAYA
        assert generator.prefix == "test_analysis"

    def test_generator_numcosmo_type(
        self, tmp_path: Path, _mock_cosmo_spec: MagicMock
    ) -> None:
        """Test NumCosmo generator initialization."""
        generator = get_generator(
            Frameworks.NUMCOSMO,
            tmp_path,
            "test_analysis",
            True,
            _mock_cosmo_spec,
            FrameworkCosmology.NONLINEAR,
        )
        assert generator.framework == Frameworks.NUMCOSMO
        assert generator.prefix == "test_analysis"
