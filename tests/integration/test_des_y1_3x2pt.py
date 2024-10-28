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
                cosmosis cosmosis/factory.ini
                cosmosis cosmosis/factory_PT.ini
                cosmosis cosmosis/default_factory.ini
                cosmosis cosmosis/pure_ccl_default_factory.ini
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
                mkdir -p numcosmo
                cd numcosmo

                numcosmo from-cosmosis ../cosmosis/factory.ini \\
                    --matter-ps eisenstein_hu \\
                    --nonlin-matter-ps halofit
                numcosmo run test factory.yaml

                numcosmo from-cosmosis ../cosmosis/factory_PT.ini \\
                    --matter-ps eisenstein_hu \\
                    --nonlin-matter-ps halofit
                numcosmo run test factory_PT.yaml

                numcosmo from-cosmosis ../cosmosis/default_factory.ini \\
                    --matter-ps eisenstein_hu \\
                    --nonlin-matter-ps halofit
                numcosmo run test default_factory.yaml

                numcosmo from-cosmosis ../cosmosis/pure_ccl_default_factory.ini \\
                    --matter-ps eisenstein_hu \\
                    --nonlin-matter-ps halofit
                numcosmo run test pure_ccl_default_factory.yaml
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
                cobaya-run cobaya/evaluate.yaml
                cobaya-run cobaya/evaluate_PT.yaml
                cobaya-run cobaya/evaluate_pure_ccl.yaml
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)
