"""Integration tests for cosmic shear"""

import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest

from firecrown.app.analysis import Frameworks
from firecrown.app.examples import ExampleCosmicShear


@pytest.fixture(name="target_framework", params=Frameworks)
def fixture_target_framework(request: pytest.FixtureRequest) -> Frameworks:
    """Generate a target framework for all frameworks."""
    return request.param


@pytest.fixture(name="use_absolute_path", params=[True, False])
def fixture_use_absolute_path(request: pytest.FixtureRequest) -> bool:
    """Generate a use_absolute_path for all path types."""
    return request.param


@pytest.fixture(name="cosmic_shear_example")
def fixture_cosmic_shear_example(
    target_framework: Frameworks,
    use_absolute_path: bool,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> tuple[Path, Frameworks]:
    """Generate configurations or verify NumCosmo's child dtype rejection.

    :param target_framework: External framework under test.
    :param use_absolute_path: Select absolute or relative generated paths.
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
        ExampleCosmicShear(
            output_path=tmp_path,
            prefix="cosmic_shear",
            target_framework=target_framework,
            use_absolute_path=use_absolute_path,
        )

    if target_framework == Frameworks.NUMCOSMO:
        captured = capfd.readouterr()
        assert "integer dtype" in captured.out + captured.err
    return tmp_path, target_framework


@pytest.mark.example
def test_cosmic_shear_run(cosmic_shear_example: tuple[Path, Frameworks]) -> None:
    """Reject the example's fractional sampling at the exact support boundary.

    :param cosmic_shear_example: Generated configuration or checked NumCosmo error.
    """
    output_path, framework = cosmic_shear_example
    match framework:
        case Frameworks.COSMOSIS:
            result = subprocess.run(
                ["cosmosis", "cosmosis_cosmic_shear.ini"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=False,
            )

            assert result.returncode != 0
            assert "integer dtype" in result.stdout + result.stderr
        case Frameworks.COBAYA:
            result = subprocess.run(
                ["cobaya-run", "--no-mpi", "-f", "cobaya_cosmic_shear.yaml"],
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
