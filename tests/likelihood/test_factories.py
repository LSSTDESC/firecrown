"""Tests for the module firecrown.likelihood.factories.
"""

import re
from pathlib import Path
import pytest

import sacc

from firecrown.likelihood.factories import (
    build_two_point_likelihood,
    DataSourceSacc,
    TwoPointCorrelationSpace,
    TwoPointExperiment,
    TwoPointFactory,
)
from firecrown.likelihood.weak_lensing import WeakLensingFactory
from firecrown.likelihood.number_counts import NumberCountsFactory
from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools


def test_two_point_factory_dict() -> None:
    two_point_factory_dict = {
        "correlation_space": TwoPointCorrelationSpace.HARMONIC,
        "weak_lensing_factory": {"per_bin_systematics": [], "global_systematics": []},
        "number_counts_factory": {"per_bin_systematics": [], "global_systematics": []},
    }
    two_point_factory = TwoPointFactory.model_validate(two_point_factory_dict)
    assert isinstance(two_point_factory, TwoPointFactory)
    assert isinstance(two_point_factory.weak_lensing_factory, WeakLensingFactory)
    assert isinstance(two_point_factory.number_counts_factory, NumberCountsFactory)
    assert two_point_factory.correlation_space == TwoPointCorrelationSpace.HARMONIC
    assert two_point_factory.weak_lensing_factory.per_bin_systematics == []
    assert two_point_factory.weak_lensing_factory.global_systematics == []
    assert two_point_factory.number_counts_factory.per_bin_systematics == []
    assert two_point_factory.number_counts_factory.global_systematics == []


def test_two_point_factor_direct() -> None:
    two_point_factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        weak_lensing_factory=WeakLensingFactory(
            per_bin_systematics=[], global_systematics=[]
        ),
        number_counts_factory=NumberCountsFactory(
            per_bin_systematics=[], global_systematics=[]
        ),
    )

    assert two_point_factory.correlation_space == TwoPointCorrelationSpace.HARMONIC
    assert two_point_factory.weak_lensing_factory.per_bin_systematics == []
    assert two_point_factory.weak_lensing_factory.global_systematics == []
    assert two_point_factory.number_counts_factory.per_bin_systematics == []
    assert two_point_factory.number_counts_factory.global_systematics == []


def test_two_point_factor_invalid_correlation_space_type() -> None:
    two_point_factory_dict = {
        "correlation_space": 1.2,
        "weak_lensing_factory": {"per_bin_systematics": [], "global_systematics": []},
        "number_counts_factory": {"per_bin_systematics": [], "global_systematics": []},
    }
    with pytest.raises(
        ValueError,
        match=re.compile(
            ".*validation error for TwoPointFactory.*correlation_space.*", re.DOTALL
        ),
    ):
        TwoPointFactory.model_validate(two_point_factory_dict)


def test_two_point_factor_invalid_correlation_space_option() -> None:
    two_point_factory_dict = {
        "correlation_space": "invalid",
        "weak_lensing_factory": {"per_bin_systematics": [], "global_systematics": []},
        "number_counts_factory": {"per_bin_systematics": [], "global_systematics": []},
    }
    with pytest.raises(
        ValueError,
        match=re.compile(
            ".*validation error for TwoPointFactory.*correlation_space.*", re.DOTALL
        ),
    ):
        TwoPointFactory.model_validate(two_point_factory_dict)


def test_data_source_sacc_dict() -> None:
    data_source_sacc_dict = {"sacc_data_file": "tests/bug_398.sacc.gz"}
    data_source_sacc = DataSourceSacc.model_validate(data_source_sacc_dict)
    assert isinstance(data_source_sacc, DataSourceSacc)
    assert data_source_sacc.sacc_data_file == "tests/bug_398.sacc.gz"


def test_data_source_sacc_direct() -> None:
    data_source_sacc = DataSourceSacc(sacc_data_file="tests/bug_398.sacc.gz")
    assert data_source_sacc.sacc_data_file == "tests/bug_398.sacc.gz"


def test_data_source_sacc_invalid_file() -> None:
    data_source_sacc_dict = {"sacc_data_file": "tests/file_not_found.sacc.gz"}
    data_source = DataSourceSacc.model_validate(data_source_sacc_dict)
    with pytest.raises(
        FileNotFoundError, match=".*File tests/file_not_found.sacc.gz does not exist.*"
    ):
        _ = data_source.get_sacc_data()


def test_data_source_sacc_get_sacc_data() -> None:
    data_source_sacc_dict = {"sacc_data_file": "tests/bug_398.sacc.gz"}
    data_source_sacc = DataSourceSacc.model_validate(data_source_sacc_dict)
    assert isinstance(data_source_sacc, DataSourceSacc)
    assert data_source_sacc.sacc_data_file == "tests/bug_398.sacc.gz"
    sacc_data = data_source_sacc.get_sacc_data()
    assert sacc_data is not None
    assert isinstance(sacc_data, sacc.Sacc)


def test_two_point_experiment_dict() -> None:
    two_point_experiment_dict = {
        "two_point_factory": {
            "correlation_space": TwoPointCorrelationSpace.HARMONIC,
            "weak_lensing_factory": {
                "per_bin_systematics": [],
                "global_systematics": [],
            },
            "number_counts_factory": {
                "per_bin_systematics": [],
                "global_systematics": [],
            },
        },
        "data_source": {"sacc_data_file": "tests/bug_398.sacc.gz"},
    }
    two_point_experiment = TwoPointExperiment.model_validate(two_point_experiment_dict)
    assert isinstance(two_point_experiment, TwoPointExperiment)
    assert isinstance(two_point_experiment.two_point_factory, TwoPointFactory)
    assert isinstance(two_point_experiment.data_source, DataSourceSacc)
    assert (
        two_point_experiment.two_point_factory.correlation_space
        == TwoPointCorrelationSpace.HARMONIC
    )
    assert (
        two_point_experiment.two_point_factory.weak_lensing_factory.per_bin_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.weak_lensing_factory.global_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.number_counts_factory.per_bin_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.number_counts_factory.global_systematics
        == []
    )
    assert two_point_experiment.data_source.sacc_data_file == "tests/bug_398.sacc.gz"


def test_two_point_experiment_direct() -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=TwoPointFactory(
            correlation_space=TwoPointCorrelationSpace.HARMONIC,
            weak_lensing_factory=WeakLensingFactory(
                per_bin_systematics=[], global_systematics=[]
            ),
            number_counts_factory=NumberCountsFactory(
                per_bin_systematics=[], global_systematics=[]
            ),
        ),
        data_source=DataSourceSacc(sacc_data_file="tests/bug_398.sacc.gz"),
    )
    assert isinstance(two_point_experiment, TwoPointExperiment)
    assert isinstance(two_point_experiment.two_point_factory, TwoPointFactory)
    assert isinstance(two_point_experiment.data_source, DataSourceSacc)
    assert (
        two_point_experiment.two_point_factory.correlation_space
        == TwoPointCorrelationSpace.HARMONIC
    )
    assert (
        two_point_experiment.two_point_factory.weak_lensing_factory.per_bin_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.weak_lensing_factory.global_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.number_counts_factory.per_bin_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.number_counts_factory.global_systematics
        == []
    )
    assert two_point_experiment.data_source.sacc_data_file == "tests/bug_398.sacc.gz"


def test_build_two_point_likelihood_real(tmp_path: Path) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    tmp_experiment_file.write_text(
        """
two_point_factory:
  correlation_space: real
  weak_lensing_factory:
    per_bin_systematics: []
    global_systematics: []
  number_counts_factory:
    per_bin_systematics: []
    global_systematics: []
data_source:
    sacc_data_file: examples/des_y1_3x2pt/sacc_data.fits
"""
    )

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    likelihood, tools = build_two_point_likelihood(build_parameters)
    assert isinstance(likelihood, Likelihood)
    assert isinstance(tools, ModelingTools)


def test_build_two_point_likelihood_harmonic(tmp_path: Path) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    tmp_experiment_file.write_text(
        """
two_point_factory:
  correlation_space: harmonic
  weak_lensing_factory:
    per_bin_systematics: []
    global_systematics: []
  number_counts_factory:
    per_bin_systematics: []
    global_systematics: []
data_source:
    sacc_data_file: tests/bug_398.sacc.gz
"""
    )

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    likelihood, tools = build_two_point_likelihood(build_parameters)
    assert isinstance(likelihood, Likelihood)
    assert isinstance(tools, ModelingTools)


def test_build_two_point_likelihood_real_no_real_data(tmp_path: Path) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    tmp_experiment_file.write_text(
        """
two_point_factory:
  correlation_space: real
  weak_lensing_factory:
    per_bin_systematics: []
    global_systematics: []
  number_counts_factory:
    per_bin_systematics: []
    global_systematics: []
data_source:
    sacc_data_file: tests/bug_398.sacc.gz
"""
    )

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    with pytest.raises(
        ValueError,
        match=re.compile(
            "No two-point measurements in real space found in the SACC file.", re.DOTALL
        ),
    ):
        _ = build_two_point_likelihood(build_parameters)


def test_build_two_point_likelihood_harmonic_no_harmonic_data(tmp_path: Path) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    tmp_experiment_file.write_text(
        """
two_point_factory:
    correlation_space: harmonic
    weak_lensing_factory:
        per_bin_systematics: []
        global_systematics: []
    number_counts_factory:
        per_bin_systematics: []
        global_systematics: []
data_source:
    sacc_data_file: examples/des_y1_3x2pt/sacc_data.fits
"""
    )

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    with pytest.raises(
        ValueError,
        match=re.compile(
            "No two-point measurements in harmonic space found in the SACC file.",
            re.DOTALL,
        ),
    ):
        _ = build_two_point_likelihood(build_parameters)


def test_build_two_point_likelihood_empty_likelihood_config(tmp_path: Path) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    tmp_experiment_file.write_text("")

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    with pytest.raises(ValueError, match="validation error for TwoPointExperiment"):
        _ = build_two_point_likelihood(build_parameters)


def test_build_two_point_likelihood_invalid_likelihood_config(tmp_path: Path) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    tmp_experiment_file.write_text("I'm not a valid YAML file.")

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    with pytest.raises(ValueError, match=".*validation error for TwoPointExperiment.*"):
        _ = build_two_point_likelihood(build_parameters)
