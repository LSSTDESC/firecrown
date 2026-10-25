"""Integration tests for the CMB lensing cross-correlation example."""

import subprocess
from pathlib import Path

import pytest

from firecrown.app.analysis import Frameworks
from firecrown.app.examples import ExampleCMBCross


@pytest.fixture(name="target_framework", params=Frameworks)
def fixture_target_framework(request: pytest.FixtureRequest) -> Frameworks:
    """Generate a target framework for all frameworks."""
    return request.param


@pytest.fixture(name="cmb_cross_example")
def fixture_cmb_cross_example(
    target_framework: Frameworks,
    tmp_path: Path,
) -> tuple[Path, Frameworks]:
    """Generate configurations for each supported framework.

    :param target_framework: External framework under test.
    :param tmp_path: Temporary output directory.
    :returns: Output directory and selected framework.
    """
    ExampleCMBCross(
        output_path=tmp_path,
        prefix="cmb_cross",
        n_bins=2,
        n_ell_points=5,
        n_z_points=100,
        target_framework=target_framework,
    )
    return tmp_path, target_framework


@pytest.mark.example
def test_cmb_cross_run(cmb_cross_example: tuple[Path, Frameworks]) -> None:
    """Run fractional example sampling through each framework's public route.

    :param cmb_cross_example: Generated framework configuration.
    """
    output_path, framework = cmb_cross_example
    match framework:
        case Frameworks.COSMOSIS:
            result = subprocess.run(
                ["cosmosis", "cosmosis_cmb_cross.ini"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
        case Frameworks.COBAYA:
            result = subprocess.run(
                ["cobaya-run", "--no-mpi", "-f", "cobaya_cmb_cross.yaml"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
        case Frameworks.NUMCOSMO:
            result = subprocess.run(
                ["numcosmo", "run", "test", "numcosmo_cmb_cross.yaml"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
