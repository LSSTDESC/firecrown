"""Integration tests for cosmic shear"""

import subprocess
import pytest


@pytest.mark.example
def test_cosmic_shear_cosmosis():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/cosmicshear
                python generate_cosmicshear_data.py
                cosmosis cosmicshear.ini
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)


@pytest.mark.example
def test_cosmic_shear_numcosmo():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/cosmicshear
                numcosmo from-cosmosis cosmicshear.ini \\
                    --matter-ps eisenstein_hu \\
                    --nonlin-matter-ps halofit
                numcosmo run test cosmicshear.yaml
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)
