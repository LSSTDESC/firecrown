"""Unit tests for firecrown.app.examples._cluster_counts module.

Tests ExampleClusterCounts example generator and build_likelihood execution.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from firecrown.app.analysis import (
    CCLCosmologySpec,
    FrameworkCosmology,
    Frameworks,
)
from firecrown.app.examples import EXAMPLES_LIST
from firecrown.app.examples._cluster_counts import (
    SIMULATED_SURVEY_NAME,
    ClusterDataset,
    ExampleClusterCounts,
)
from firecrown.app.examples._cluster_data import SDSS_SURVEY_NAME
from firecrown.likelihood import ConstGaussian
from firecrown.modeling_tools import ModelingTools


def make_builder(tmp_path: Path, **kwargs) -> ExampleClusterCounts:
    """Create a builder without running the generation workflow."""
    with patch.object(ExampleClusterCounts, "_proceed_generation"):
        return ExampleClusterCounts(
            output_path=tmp_path, target_framework=Frameworks.COSMOSIS, **kwargs
        )


def test_registered() -> None:
    """The example is available from the command line."""
    assert EXAMPLES_LIST["cluster_counts"] is ExampleClusterCounts


def test_defaults(tmp_path: Path) -> None:
    """Default options fit counts and mean mass of the simulated catalog."""
    builder = make_builder(tmp_path)

    assert builder.prefix == "cluster_counts"
    assert builder.dataset == ClusterDataset.NUMCOSMO
    assert builder.use_cluster_counts is True
    assert builder.use_mean_log_mass is True
    assert builder.seed == 32


def test_requires_an_observable(tmp_path: Path) -> None:
    """Disabling both observables is rejected."""
    with pytest.raises(ValueError, match="At least one of"):
        make_builder(tmp_path, use_cluster_counts=False, use_mean_log_mass=False)


@pytest.mark.parametrize(
    "dataset,survey_name",
    [
        (ClusterDataset.NUMCOSMO, SIMULATED_SURVEY_NAME),
        (ClusterDataset.SDSS, SDSS_SURVEY_NAME),
    ],
)
def test_get_build_parameters(
    tmp_path: Path, dataset: ClusterDataset, survey_name: str
) -> None:
    """Build parameters carry the SACC file, survey and observables."""
    builder = make_builder(tmp_path, dataset=dataset, use_mean_log_mass=False)

    params = builder.get_build_parameters(tmp_path / "data.sacc")

    assert params.get_string("sacc_file") == (tmp_path / "data.sacc").as_posix()
    assert params.get_string("survey_name") == survey_name
    assert params.get_bool("use_cluster_counts") is True
    assert params.get_bool("use_mean_log_mass") is False


def test_get_models(tmp_path: Path) -> None:
    """The six mass-richness parameters are free."""
    models = make_builder(tmp_path, prefix="cc").get_models()

    assert len(models) == 1
    assert models[0].name == "firecrown_cc"
    assert len(models[0].parameters) == 6
    assert all(p.free for p in models[0].parameters)
    assert "mass_distribution_mu0" in models[0]


def test_cosmology(tmp_path: Path) -> None:
    """Linear power spectrum and the simulation cosmology are required."""
    builder = make_builder(tmp_path)

    assert builder.required_cosmology() == FrameworkCosmology.LINEAR
    spec = builder.cosmology_analysis_spec()
    assert spec["Omega_c"].free
    assert spec["sigma8"].free


def test_cosmology_spec_override(tmp_path: Path) -> None:
    """A user cosmology specification takes precedence."""
    spec_file = tmp_path / "cosmo.yaml"
    spec_file.write_text(
        CCLCosmologySpec.vanilla_lcdm().model_dump_json(), encoding="utf-8"
    )
    builder = make_builder(tmp_path, cosmology_spec=spec_file)

    assert builder.cosmology_analysis_spec() == CCLCosmologySpec.vanilla_lcdm()


def test_options_desc(tmp_path: Path) -> None:
    """The seed is only reported for the simulated data set."""
    sim = dict(make_builder(tmp_path).get_options_desc())
    sdss = dict(make_builder(tmp_path, dataset=ClusterDataset.SDSS).get_options_desc())

    assert sim["Data set"] == "numcosmo"
    assert "Seed" in sim
    assert sdss["Data set"] == "sdss"
    assert "Seed" not in sdss


@pytest.mark.parametrize("dataset", list(ClusterDataset))
def test_build_likelihood_execution(tmp_path: Path, dataset: ClusterDataset) -> None:
    """The generated data and factory build a likelihood."""
    builder = make_builder(tmp_path, prefix="test_cc", dataset=dataset)

    sacc_file = builder.generate_sacc(tmp_path)
    factory_file = builder.generate_factory(tmp_path, sacc_file)
    params = builder.get_build_parameters(sacc_file)

    assert sacc_file == tmp_path / "test_cc.sacc"
    assert factory_file == tmp_path / "test_cc_factory.py"

    # pylint: disable-next=import-outside-toplevel
    from firecrown.app.examples import _cluster_counts_template

    likelihood, tools = _cluster_counts_template.build_likelihood(params)
    assert isinstance(likelihood, ConstGaussian)
    assert isinstance(tools, ModelingTools)

    spec = importlib.util.spec_from_file_location("test_cc_factory", factory_file)
    assert spec is not None
    assert spec.loader is not None
    factory_module = importlib.util.module_from_spec(spec)
    sys.modules["test_cc_factory"] = factory_module
    try:
        spec.loader.exec_module(factory_module)
        likelihood_copy, _ = factory_module.build_likelihood(params)
        assert isinstance(likelihood_copy, ConstGaussian)
    finally:
        del sys.modules["test_cc_factory"]


@pytest.mark.parametrize(
    "use_cluster_counts,use_mean_log_mass", [(True, False), (False, True)]
)
def test_build_likelihood_single_observable(
    tmp_path: Path, use_cluster_counts: bool, use_mean_log_mass: bool
) -> None:
    """The factory builds a likelihood from either observable alone."""
    builder = make_builder(
        tmp_path,
        dataset=ClusterDataset.SDSS,
        use_cluster_counts=use_cluster_counts,
        use_mean_log_mass=use_mean_log_mass,
    )
    sacc_file = builder.generate_sacc(tmp_path)

    # pylint: disable-next=import-outside-toplevel
    from firecrown.app.examples import _cluster_counts_template

    likelihood, _ = _cluster_counts_template.build_likelihood(
        builder.get_build_parameters(sacc_file)
    )

    assert isinstance(likelihood, ConstGaussian)
    assert len(likelihood.get_data_vector()) == 5
