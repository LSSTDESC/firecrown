import subprocess
import pytest


@pytest.mark.integration
def test_srd_sn_cosmosis():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/srd_sn
                cosmosis sn_srd.ini
                cosmosis sn_only.ini
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)


@pytest.mark.integration
def test_srd_sn_numcosmo():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/srd_sn
                numcosmo from-cosmosis sn_srd.ini
                numcosmo run test sn_srd.yaml
                numcosmo from-cosmosis sn_only.ini
                numcosmo run test sn_only.yaml
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)
