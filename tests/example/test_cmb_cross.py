"""Integration tests for the CMB lensing cross-correlation example."""

import subprocess
from contextlib import nullcontext
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
    capfd: pytest.CaptureFixture[str],
) -> tuple[Path, Frameworks]:
    """Generate configurations or verify NumCosmo's child dtype rejection.

    :param target_framework: External framework under test.
    :param tmp_path: Temporary output directory.
    :param capfd: Capture the NumCosmo configuration subprocess diagnostic.
    :returns: Output directory and selected framework.
    """
    # NumCosmo reads the likelihood while creating its configuration. The
    # other frameworks reach the same exact-multipole boundary at execution.
    context = (
        pytest.raises(RuntimeError, match="write_config.*subprocess failed")
        if target_framework == Frameworks.NUMCOSMO
        else nullcontext()
    )
    with context:
        ExampleCMBCross(
            output_path=tmp_path,
            prefix="cmb_cross",
            n_bins=2,
            n_ell_points=5,
            n_z_points=100,
            target_framework=target_framework,
        )

    if target_framework == Frameworks.NUMCOSMO:
        captured = capfd.readouterr()
        assert "integer dtype" in captured.out + captured.err
    return tmp_path, target_framework


@pytest.mark.example
def test_cmb_cross_run(cmb_cross_example: tuple[Path, Frameworks]) -> None:
    """Reject fractional example support through each framework's public route.

    These real sampling coordinates are retained. No new scientific rounding
    is chosen to make them satisfy exact integer multipoles.

    :param cmb_cross_example: Generated framework configuration or checked error.
    """
    output_path, framework = cmb_cross_example
    match framework:
        case Frameworks.COSMOSIS:
            result = subprocess.run(
                ["cosmosis", "cosmosis_cmb_cross.ini"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=False,
            )

            assert result.returncode != 0
            assert "integer dtype" in result.stdout + result.stderr
        case Frameworks.COBAYA:
            result = subprocess.run(
                ["cobaya-run", "--no-mpi", "-f", "cobaya_cmb_cross.yaml"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=False,
            )

            assert result.returncode != 0
            assert "integer dtype" in result.stdout + result.stderr
        case Frameworks.NUMCOSMO:
            # The fixture verifies the child dtype error during configuration.
            return
