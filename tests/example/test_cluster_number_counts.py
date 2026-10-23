"""Integration tests for the galaxy cluster examples."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from firecrown.app.analysis import AnalysisBuilder, Frameworks
from firecrown.app.examples import (
    ClusterDataset,
    ClusterProfile,
    ExampleClusterCounts,
    ExampleClusterShear,
)

CLUSTER_CASES: dict[str, tuple[type[AnalysisBuilder], dict[str, Any]]] = {
    "counts_mean_mass": (ExampleClusterCounts, {}),
    "counts": (ExampleClusterCounts, {"use_mean_log_mass": False}),
    "mean_mass": (ExampleClusterCounts, {"use_cluster_counts": False}),
    "sdss": (ExampleClusterCounts, {"dataset": ClusterDataset.SDSS}),
    "delta_sigma": (ExampleClusterShear, {"profile": ClusterProfile.DELTA_SIGMA}),
    "reduced_shear": (
        ExampleClusterShear,
        {"profile": ClusterProfile.REDUCED_SHEAR},
    ),
}


@pytest.fixture(name="target_framework", params=Frameworks)
def fixture_target_framework(request) -> Frameworks:
    """Generate a target framework for all frameworks."""
    return request.param


@pytest.fixture(name="cluster_example", params=CLUSTER_CASES)
def fixture_cluster_example(
    request, target_framework: Frameworks, tmp_path: Path
) -> tuple[Path, Frameworks]:
    """Generate each cluster example for all frameworks."""
    example_cls, options = CLUSTER_CASES[request.param]
    example_cls(
        output_path=tmp_path,
        prefix="cluster",
        target_framework=target_framework,
        **options,
    )
    return tmp_path, target_framework


@pytest.mark.example
def test_cluster_run(cluster_example: tuple[Path, Frameworks]) -> None:
    """Run each framework's generated configuration end-to-end."""
    output_path, framework = cluster_example
    match framework:
        case Frameworks.COSMOSIS:
            command = ["cosmosis", "cosmosis_cluster.ini"]
        case Frameworks.COBAYA:
            command = ["cobaya-run", "--no-mpi", "-f", "cobaya_cluster.yaml"]
        case Frameworks.NUMCOSMO:
            command = ["numcosmo", "run", "test", "numcosmo_cluster.yaml"]

    result = subprocess.run(
        command, cwd=output_path, capture_output=True, text=True, check=True
    )
    assert result.returncode == 0
