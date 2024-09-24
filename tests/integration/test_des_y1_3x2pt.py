"""Integration tests for the DES Y1 3x2pt analysis"""

import subprocess
import pytest


@pytest.mark.integration
def test_des_y1_3x2pt_cosmosis():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/des_y1_3x2pt
                cosmosis des_y1_3x2pt.ini
                cosmosis des_y1_3x2pt_PT.ini
                cosmosis des_y1_3x2pt_default_factory.ini
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)


@pytest.mark.integration
def test_des_y1_3x2pt_numcosmo():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/des_y1_3x2pt
                numcosmo from-cosmosis des_y1_3x2pt.ini --matter-ps eisenstein_hu\\
                    --nonlin-matter-ps halofit
                numcosmo run test des_y1_3x2pt.yaml
                numcosmo from-cosmosis des_y1_3x2pt_PT.ini --matter-ps eisenstein_hu\\
                    --nonlin-matter-ps halofit
                numcosmo run test des_y1_3x2pt_PT.yaml
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)


@pytest.mark.integration
def test_des_y1_3x2pt_cobaya():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/des_y1_3x2pt
                cobaya-run cobaya_evaluate.yaml
                cobaya-run cobaya_evaluate_PT.yaml
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)
