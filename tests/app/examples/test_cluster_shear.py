"""Unit tests for firecrown.app.examples._cluster_shear module.

Tests ExampleClusterShear example generator and build_likelihood execution.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from firecrown.app.analysis import FrameworkCosmology, Frameworks
from firecrown.app.examples import EXAMPLES_LIST
from firecrown.app.examples import _cluster_shear_template
from firecrown.app.examples._cluster_data import SIM_Z_RANGE, ClusterCatalog
from firecrown.app.examples._cluster_shear import (
    SIMULATED_SURVEY_NAME,
    ClusterProfile,
    ExampleClusterShear,
)
from firecrown.likelihood import ConstGaussian, NamedParameters
from firecrown.modeling_tools import ModelingTools


def make_builder(tmp_path: Path, **kwargs) -> ExampleClusterShear:
    """Create a builder without running the generation workflow."""
    with patch.object(ExampleClusterShear, "_proceed_generation"):
        return ExampleClusterShear(
            output_path=tmp_path, target_framework=Frameworks.COSMOSIS, **kwargs
        )


def small_catalog(_seed: int, _mass_def: object) -> ClusterCatalog:
    """Replace simulate_catalog with a uniform catalog that fills every bin."""
    rng = np.random.default_rng(0)
    n_clusters = 400
    return ClusterCatalog(
        z=rng.uniform(*SIM_Z_RANGE, n_clusters),
        log10_richness=rng.uniform(1.0, 2.0, n_clusters),
        log10_mass=rng.normal(14.0, 0.3, n_clusters),
    )


@pytest.fixture(name="sacc_file", scope="module")
def fixture_sacc_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """SACC file generated from a small catalog."""
    tmp_path = tmp_path_factory.mktemp("cluster_shear")
    builder = make_builder(tmp_path)
    with patch("firecrown.app.examples._cluster_shear.simulate_catalog", small_catalog):
        return builder.generate_sacc(tmp_path)


def test_registered() -> None:
    """The example is available from the command line."""
    assert EXAMPLES_LIST["cluster_shear"] is ExampleClusterShear


def test_defaults(tmp_path: Path) -> None:
    """Default options fit counts and DeltaSigma."""
    builder = make_builder(tmp_path)

    assert builder.prefix == "cluster_shear"
    assert builder.profile == ClusterProfile.DELTA_SIGMA
    assert builder.use_cluster_counts is True
    assert builder.use_mean_log_mass is False
    assert builder.seed == 2


def test_get_build_parameters(tmp_path: Path) -> None:
    """Build parameters carry the SACC file, survey, profile and observables."""
    builder = make_builder(tmp_path, profile=ClusterProfile.REDUCED_SHEAR)

    params = builder.get_build_parameters(tmp_path / "data.sacc")

    assert params.get_string("survey_name") == SIMULATED_SURVEY_NAME
    assert params.get_string("profile") == "reduced_shear"
    assert params.get_bool("use_cluster_counts") is True
    assert params.get_bool("use_mean_log_mass") is False


def test_get_models(tmp_path: Path) -> None:
    """Mass-richness and concentration parameters are all fixed."""
    models = make_builder(tmp_path).get_models()

    assert len(models) == 1
    assert len(models[0].parameters) == 7
    assert not any(p.free for p in models[0].parameters)
    assert models[0]["cluster_theory_cluster_concentration"].default_value == 4.0


def test_cosmology_and_options(tmp_path: Path) -> None:
    """Linear power spectrum is required; options are described."""
    builder = make_builder(tmp_path)

    assert builder.required_cosmology() == FrameworkCosmology.LINEAR
    assert builder.cosmology_analysis_spec()["sigma8"].free
    assert dict(builder.get_options_desc())["Lensing profile"] == "delta_sigma"


@pytest.mark.parametrize("profile", list(ClusterProfile))
def test_build_likelihood_execution(
    tmp_path: Path, sacc_file: Path, profile: ClusterProfile
) -> None:
    """The generated data and factory build a likelihood for each profile."""
    builder = make_builder(tmp_path, prefix="test_cs", profile=profile)

    factory_file = builder.generate_factory(tmp_path, sacc_file)
    assert factory_file == tmp_path / "test_cs_factory.py"
    assert factory_file.exists()

    params = builder.get_build_parameters(sacc_file)
    likelihood, tools = _cluster_shear_template.build_likelihood(params)

    assert isinstance(likelihood, ConstGaussian)
    assert isinstance(tools, ModelingTools)
    assert len(likelihood.statistics) == 2


def test_build_likelihood_unknown_profile(sacc_file: Path) -> None:
    """An unknown profile is rejected."""
    params = NamedParameters(
        {
            "sacc_file": sacc_file.as_posix(),
            "survey_name": SIMULATED_SURVEY_NAME,
            "profile": "convergence",
        }
    )
    with pytest.raises(ValueError, match="Unknown lensing profile"):
        _cluster_shear_template.build_likelihood(params)
