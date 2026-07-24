"""Integration tests for the CMB lensing cross-correlation example."""

import subprocess
from pathlib import Path

import pytest

from firecrown.app.analysis import Frameworks
from firecrown.app.examples import ExampleCMBCross


@pytest.fixture(name="target_framework", params=Frameworks)
def fixture_target_framework(request) -> Frameworks:
    """Generate a target framework for all frameworks."""
    return request.param


@pytest.fixture(name="cmb_cross_example")
def fixture_cmb_cross_example(
    target_framework: Frameworks, tmp_path: Path
) -> tuple[Path, Frameworks]:
    """Generate the CMB cross-correlation example for all frameworks."""
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
def test_cmb_cross_run(cmb_cross_example):
    """Run each framework's generated configuration end-to-end.

    This is what actually exercises the background/distance splines out to
    the CMB lensing source redshift (z~1100); a framework whose pipeline
    doesn't reach far enough would fail here with a ValueError, even though
    the SACC/likelihood-construction unit tests pass.
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
                ["cobaya-run", "-f", "cobaya_cmb_cross.yaml"],
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
