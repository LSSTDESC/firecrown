"""Integration tests for the DES Y1 3x2pt analysis"""

import subprocess
import pytest

INI_FILES = [
    "factory.ini",
    "factory_PT.ini",
    "default_factory.ini",
    "pure_ccl.ini",
    "mu_sigma.ini",
]

COBAYA_YAML_FILES = [
    "evaluate.yaml",
    "evaluate_PT.yaml",
    "evaluate_pure_ccl.yaml",
    "evaluate_mu_sigma.yaml",
]


@pytest.fixture(name="ini_file", params=INI_FILES)
def fixture_ini_file(request) -> str:
    """Fixture to provide the ini files for the DES Y1 3x2pt analysis."""
    return request.param


@pytest.fixture(name="cobaya_yaml_file", params=COBAYA_YAML_FILES)
def fixture_cobaya_yaml_file(request) -> str:
    """Fixture to provide the cobaya yaml files for the DES Y1 3x2pt analysis."""
    return request.param


@pytest.mark.integration
def test_des_y1_3x2pt_cosmosis(ini_file: str):
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
                set -e
                cd examples/des_y1_3x2pt
                cosmosis cosmosis/{ini_file}
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)


@pytest.mark.integration
def test_des_y1_3x2pt_numcosmo(ini_file: str):
    # FIXME: New NumCosmo 0.23.0 on will support the mu_sigma.ini. Update the following
    # line to remove the skip when the new version is released.
    if "mu_sigma" in ini_file:
        pytest.skip("numcosmo does not support mu_sigma.ini")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
                set -e
                cd examples/des_y1_3x2pt
                mkdir -p numcosmo
                cd numcosmo

                numcosmo from-cosmosis ../cosmosis/{ini_file} \\
                    --matter-ps eisenstein_hu \\
                    --nonlin-matter-ps halofit
                numcosmo run test {ini_file.replace('.ini', '.yaml')}
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)


@pytest.mark.integration
def test_des_y1_3x2pt_cobaya(cobaya_yaml_file: str):
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
                set -e
                cd examples/des_y1_3x2pt
                cobaya-run cobaya/{cobaya_yaml_file}
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)
