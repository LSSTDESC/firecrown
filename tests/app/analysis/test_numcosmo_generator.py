"""Unit tests for NumCosmoConfigGenerator class.

Tests for the main generator class and related setup functions
in firecrown.app.analysis._numcosmo module.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from numcosmo_py import Ncm, Nc

from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._numcosmo import (
    ConfigOptions,
    NAME_MAP,
    NumCosmoConfigGenerator,
    _set_standard_params,
)
from firecrown.app.analysis._types import (
    Frameworks,
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


@pytest.fixture(name="minimal_factory_file")
def fixture_minimal_factory_file(tmp_path: Path) -> Path:
    """Create a minimal factory file for testing.

    This factory creates a simple likelihood with:
    - One tracer (lens0)
    - One two-point statistic (galaxy_density_cl)
    - Mock SACC data with 3 data points
    """
    factory_file = tmp_path / "factory.py"
    factory_file.write_text("""
import sacc
import numpy as np
from firecrown.likelihood.number_counts import NumberCounts
from firecrown.likelihood import ConstGaussian, TwoPoint


def build_likelihood(_):
    lens0 = NumberCounts(sacc_tracer="lens0")
    two_point = TwoPoint("galaxy_density_cl", source0=lens0, source1=lens0)
    statistics = [two_point]

    sacc_data = sacc.Sacc()
    sacc_data.add_tracer(
        "NZ", "lens0", np.array([0.1, 0.2, 0.3]), np.array([0.0, 1.0, 0.0])
    )
    sacc_data.add_ell_cl(
        "galaxy_density_cl",
        "lens0",
        "lens0",
        np.array([10, 20, 30]),
        np.array([1.0, 2.0, 3.0]),
    )
    sacc_data.add_covariance(np.eye(3) * 0.1)

    likelihood = ConstGaussian(statistics=statistics)
    likelihood.read(sacc_data)
    return likelihood
""".strip())
    return factory_file


class TestNumCosmoConfigGenerator:
    """Tests for NumCosmoConfigGenerator class."""

    def test_generator_framework(self) -> None:
        """Test that generator has correct framework."""
        assert NumCosmoConfigGenerator.framework == Frameworks.NUMCOSMO

    def test_generator_initialization(
        self, numcosmo_init: bool, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test generator initialization."""
        assert numcosmo_init
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )
        assert gen.framework == Frameworks.NUMCOSMO
        assert gen.prefix == "test"
        assert gen.output_path == tmp_path

    def test_generator_write_config(
        self,
        numcosmo_init: bool,
        tmp_path: Path,
        vanilla_cosmo: CCLCosmologySpec,
        minimal_factory_file: Path,
    ) -> None:
        """Test that write_config creates YAML files."""
        assert numcosmo_init
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="my_analysis",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})
        gen.write_config()

        # Check that files were created
        expected_file = tmp_path / "numcosmo_my_analysis.yaml"
        assert expected_file.exists()

    def test_generator_write_config_subprocess_failure(
        self,
        numcosmo_init: bool,
        tmp_path: Path,
        vanilla_cosmo: CCLCosmologySpec,
        minimal_factory_file: Path,
    ) -> None:
        """Test that write_config raises RuntimeError when subprocess fails.

        This test patches the multiprocessing.Process to simulate a subprocess
        failure by setting exitcode to a non-zero value. This triggers the
        RuntimeError at line 658 of _numcosmo.py.
        """
        assert numcosmo_init
        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="my_analysis",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        gen.factory_source = minimal_factory_file
        gen.build_parameters = NamedParameters({})

        # Create a mock process that simulates a failure
        mock_process = MagicMock()
        mock_process.exitcode = 1  # Non-zero exit code

        # Patch the Process class to return our mock
        with patch("multiprocessing.get_context") as mock_get_context:
            mock_ctx = MagicMock()
            mock_ctx.Process.return_value = mock_process
            mock_get_context.return_value = mock_ctx

            # This should raise RuntimeError due to non-zero exit code
            with pytest.raises(
                RuntimeError,
                match=r"write_config\(\) subprocess failed with exit code 1",
            ):
                gen.write_config()


class TestSetStandardParams:
    """Tests for _set_standard_params function error handling."""

    def test_set_standard_params_unknown_parameter(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test that ValueError is raised when parameter is not found in models.

        This test patches NAME_MAP to point a parameter to a nonexistent NumCosmo
        parameter name. This causes the for-else loop to fail to find the parameter
        in any model, raising ValueError with "Unknown parameter" message
        (line 266 of _numcosmo.py).
        """
        assert numcosmo_init

        # Create a standard cosmology spec
        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()

        # Create minimal config options
        config_opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create minimal NumCosmo objects
        mset = Ncm.MSet.new_array(
            [Nc.HICosmoDECpl.new()]  # pylint: disable=no-value-for-parameter
        )
        cosmo = mset.get(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
        reion = Nc.HIReionCamb.new()  # pylint: disable=no-value-for-parameter
        priors: list[Ncm.Prior] = []

        assert isinstance(cosmo, Nc.HICosmoDECpl)

        # Patch NAME_MAP to map a valid parameter to a nonexistent NumCosmo param
        with patch.dict(
            NAME_MAP,
            {"Omega_c": "nonexistent_param_in_cosmology"},
        ):
            # This should raise ValueError because "nonexistent_param_in_cosmology"
            # is not in cosmo.param_names(), prim.param_names(), or reion.param_names()
            with pytest.raises(ValueError, match="Unknown parameter Omega_c"):
                _set_standard_params(config_opts, mset, cosmo, prim, reion, priors)
