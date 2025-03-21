"""Integration tests for cluster number counts"""

import subprocess
import pytest


@pytest.mark.integration
def test_cluster_number_counts_cosmosis():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/cluster_number_counts
                python generate_rich_mean_mass_sacc_data.py
                cosmosis cluster_counts_redshift_richness.ini
                cosmosis cluster_mean_mass_redshift_richness.ini
                cosmosis cluster_counts_mean_mass_redshift_richness.ini
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)


@pytest.mark.integration
def test_cluster_number_counts_sdss_cosmosis():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/cluster_number_counts
                python generate_SDSS_ClusterCountsMass_sacc_data.py
                cosmosis cluster_SDSS_counts_mean_mass_redshift_richness.ini
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)


@pytest.mark.integration
def test_cluster_number_counts_numcosmo():
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
                set -e
                cd examples/cluster_number_counts
                numcosmo from-cosmosis cluster_counts_redshift_richness.ini\\
                    --matter-ps eisenstein_hu
                numcosmo run test cluster_counts_redshift_richness.yaml
                numcosmo from-cosmosis cluster_mean_mass_redshift_richness.ini\\
                    --matter-ps eisenstein_hu
                numcosmo run test cluster_mean_mass_redshift_richness.yaml
                numcosmo from-cosmosis cluster_counts_mean_mass_redshift_richness.ini\\
                    --matter-ps eisenstein_hu
                numcosmo run test cluster_counts_mean_mass_redshift_richness.yaml
                numcosmo from-cosmosis cluster_SDSS_counts_mean_mass_redshift_richness.ini\\
                    --matter-ps eisenstein_hu
                numcosmo run test cluster_SDSS_counts_mean_mass_redshift_richness.yaml
            """,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    print(result.stdout)
