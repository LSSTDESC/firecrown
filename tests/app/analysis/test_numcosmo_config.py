"""Unit tests for NumCosmo configuration handling.

Tests for ConfigOptions dataclass and _create_mapping function
in firecrown.app.analysis._numcosmo module.
"""

from pathlib import Path
import pytest

from numcosmo_py import Ncm

from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._numcosmo import (
    ConfigOptions,
    _create_mapping,
)
from firecrown.app.analysis._types import (
    FrameworkCosmology,
    CCLCosmologySpec,
)


@pytest.fixture(name="numcosmo_init", scope="session")
def fixture_numcosmo_init() -> bool:
    """Fixture to initialize NumCosmo for testing."""
    Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

    return True


@pytest.fixture(name="vanilla_cosmo")
def fixture_vanilla_cosmo() -> CCLCosmologySpec:
    """Create vanilla LCDM cosmology spec for testing."""
    return CCLCosmologySpec.vanilla_lcdm()


@pytest.fixture(name="config_options")
def fixture_config_options(
    tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
) -> ConfigOptions:
    """Create ConfigOptions for testing."""
    return ConfigOptions(
        output_path=tmp_path,
        factory_source=Path("factory.py"),
        build_parameters=NamedParameters({}),
        models=[],
        cosmo_spec=vanilla_cosmo,
        use_absolute_path=True,
        required_cosmology=FrameworkCosmology.NONLINEAR,
        prefix="test",
    )


class TestConfigOptions:
    """Tests for ConfigOptions dataclass."""

    def test_config_options_initialization(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test ConfigOptions initialization."""
        opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        assert opts.output_path == tmp_path
        assert opts.prefix == "test"
        assert opts.required_cosmology == FrameworkCosmology.NONLINEAR
        assert opts.distance_max_z == 4.0  # default
        assert opts.reltol == 1e-7  # default

    def test_config_options_custom_parameters(
        self, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test ConfigOptions with custom parameters."""
        opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=False,
            required_cosmology=FrameworkCosmology.LINEAR,
            prefix="custom",
            distance_max_z=5.0,
            reltol=1e-8,
        )

        assert opts.distance_max_z == 5.0
        assert opts.reltol == 1e-8
        assert opts.required_cosmology == FrameworkCosmology.LINEAR


class TestCreateMapping:
    """Tests for _create_mapping function."""

    def test_create_mapping_none_cosmology(self, config_options: ConfigOptions) -> None:
        """Test that NONE cosmology returns None."""
        config_options.required_cosmology = FrameworkCosmology.NONE
        result = _create_mapping(config_options)
        assert result is None

    def test_create_mapping_nonlinear(
        self, numcosmo_init: bool, config_options: ConfigOptions
    ) -> None:
        """Test creating mapping for nonlinear cosmology."""
        assert numcosmo_init
        config_options.required_cosmology = FrameworkCosmology.NONLINEAR
        result = _create_mapping(config_options)
        assert result is not None

    def test_create_mapping_linear(
        self, numcosmo_init: bool, config_options: ConfigOptions
    ) -> None:
        """Test creating mapping for linear cosmology."""
        assert numcosmo_init
        config_options.required_cosmology = FrameworkCosmology.LINEAR
        result = _create_mapping(config_options)
        assert result is not None

    def test_create_mapping_background(
        self, numcosmo_init: bool, config_options: ConfigOptions
    ) -> None:
        """Test creating mapping for background cosmology."""
        assert numcosmo_init
        config_options.required_cosmology = FrameworkCosmology.BACKGROUND
        result = _create_mapping(config_options)
        assert result is not None
