"""Integration tests for the DES Y1 3x2pt analysis"""

import subprocess
from pathlib import Path
import pytest

from firecrown.app.analysis import Frameworks
from firecrown.app.examples import ExampleDESY13x2pt, DESY1FactoryType


@pytest.fixture(name="target_framework", params=Frameworks)
def fixture_target_framework(request) -> Frameworks:
    """Generate a target framework for all frameworks."""
    return request.param


@pytest.fixture(name="use_absolute_path", params=[True, False])
def fixture_use_absolute_path(request) -> bool:
    """Generate a use_absolute_path for all path types."""
    return request.param


@pytest.fixture(name="factory_type", params=DESY1FactoryType)
def fixture_factory_type(request) -> DESY1FactoryType:
    """Generate a factory type for all factory types."""
    return request.param


@pytest.fixture(name="des_y1_3x2pt_example")
def fixture_des_y1_3x2pt_example(
    target_framework: Frameworks,
    use_absolute_path: bool,
    factory_type: DESY1FactoryType,
    tmp_path: Path,
) -> tuple[Path, Frameworks]:
    """Generate DES Y1 3x2pt example for all frameworks, path types, and factory types."""
    ExampleDESY13x2pt(
        output_path=tmp_path,
        prefix="des_y1_3x2pt",
        target_framework=target_framework,
        use_absolute_path=use_absolute_path,
        factory_type=factory_type,
    )

    return tmp_path, target_framework


@pytest.mark.example
def test_des_y1_3x2pt_run(des_y1_3x2pt_example):
    output_path, framework = des_y1_3x2pt_example
    match framework:
        case Frameworks.COSMOSIS:
            result = subprocess.run(
                ["cosmosis", "cosmosis_des_y1_3x2pt.ini"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
        case Frameworks.COBAYA:
            result = subprocess.run(
                ["cobaya-run", "-f", "cobaya_des_y1_3x2pt.yaml"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
        case Frameworks.NUMCOSMO:
            result = subprocess.run(
                ["numcosmo", "run", "test", "numcosmo_des_y1_3x2pt.yaml"],
                cwd=output_path,
                capture_output=True,
                text=True,
                check=True,
            )

            assert result.returncode == 0
