"""Integration test: the supernova and cosmic shear example runs in every framework."""

import subprocess
from pathlib import Path

import pytest

from firecrown.app.analysis import Frameworks
from firecrown.app.examples import ExampleSNCosmicShear

COMMANDS = {
    Frameworks.COSMOSIS: ["cosmosis", "cosmosis_sn_cosmic_shear.ini"],
    Frameworks.COBAYA: ["cobaya-run", "--no-mpi", "-f", "cobaya_sn_cosmic_shear.yaml"],
    Frameworks.NUMCOSMO: ["numcosmo", "run", "test", "numcosmo_sn_cosmic_shear.yaml"],
}


@pytest.mark.example
@pytest.mark.parametrize("use_absolute_path", [True, False])
@pytest.mark.parametrize("framework", Frameworks)
def test_sn_cosmic_shear_run(
    framework: Frameworks, use_absolute_path: bool, tmp_path: Path
) -> None:
    """Generate the example and evaluate its likelihood once."""
    ExampleSNCosmicShear(
        output_path=tmp_path,
        prefix="sn_cosmic_shear",
        target_framework=framework,
        use_absolute_path=use_absolute_path,
    )

    result = subprocess.run(
        COMMANDS[framework], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert result.returncode == 0
