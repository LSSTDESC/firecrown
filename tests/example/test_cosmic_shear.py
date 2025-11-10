"""Integration tests for cosmic shear"""

import subprocess
from pathlib import Path
import pytest

from firecrown.app.analysis import Frameworks
from firecrown.app.examples import ExampleCosmicShear


@pytest.fixture(name="target_framework", params=Frameworks)
def fixture_target_framework(request) -> Frameworks:
    """Generate a target framework for all frameworks."""
    return request.param


@pytest.fixture(name="use_absolute_path", params=[True, False])
def fixture_use_absolute_path(request) -> bool:
    """Generate a use_absolute_path for all path types."""
    return request.param


@pytest.fixture(name="cosmic_shear_example")
def fixture_cosmic_shear_example(
    target_framework: Frameworks, use_absolute_path: bool, tmp_path: Path
) -> tuple[Path, Frameworks]:
    """Generate cosmic shear example for all frameworks and path types."""
    ExampleCosmicShear(
        output_path=tmp_path,
        prefix="cosmic_shear",
        target_framework=target_framework,
        use_absolute_path=use_absolute_path,
    )

    return tmp_path, target_framework


@pytest.mark.example
def test_cosmic_shear_run(cosmic_shear_example):
    output_path, framework = cosmic_shear_example
    match framework:
        case Frameworks.COSMOSIS:
            result = subprocess.run(
                ["cosmosis", "cosmosis_cosmic_shear.ini"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
        case Frameworks.COBAYA:
            result = subprocess.run(
                ["cobaya-run", "-f", "cobaya_cosmic_shear.yaml"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
        case Frameworks.NUMCOSMO:
            result = subprocess.run(
                ["numcosmo", "run", "test", "numcosmo_cosmic_shear.yaml"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
