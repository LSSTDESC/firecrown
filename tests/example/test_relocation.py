"""Integration test: a generated example runs after its directory is moved."""

import shutil
import subprocess
from pathlib import Path

import pytest

from firecrown.app.analysis import Frameworks
from firecrown.app.examples import ExampleCosmicShear

COMMANDS = {
    Frameworks.COSMOSIS: ["cosmosis", "cosmosis_cosmic_shear.ini"],
    Frameworks.COBAYA: ["cobaya-run", "--no-mpi", "-f", "cobaya_cosmic_shear.yaml"],
    Frameworks.NUMCOSMO: ["numcosmo", "run", "test", "numcosmo_cosmic_shear.yaml"],
}


@pytest.mark.example
@pytest.mark.parametrize("framework", Frameworks)
def test_run_after_move(framework: Frameworks, tmp_path: Path) -> None:
    """Generate the cosmic shear example, move its directory and run it there."""
    generated = tmp_path / "generated"
    ExampleCosmicShear(
        output_path=generated, prefix="cosmic_shear", target_framework=framework
    )
    moved = tmp_path / "moved" / "cosmic_shear"
    moved.parent.mkdir()
    shutil.move(generated, moved)
    assert not generated.exists()

    result = subprocess.run(
        COMMANDS[framework], cwd=moved, capture_output=True, text=True, check=True
    )
    assert result.returncode == 0
