"""Tests for the module firecrown.likelihood.factories."""

import re
from pathlib import Path
import pytest
import numpy as np

import pyccl
import sacc
from firecrown.metadata_types import TwoPointCorrelationSpace
from firecrown.likelihood.factories import (
    build_two_point_likelihood,
    DataSourceSacc,
    load_sacc_data,
    TwoPointExperiment,
)
from firecrown.likelihood.factories._sacc_utils import ensure_path
from firecrown.likelihood import TwoPoint, TwoPointFactory
from firecrown.likelihood.weak_lensing import WeakLensingFactory
from firecrown.likelihood.number_counts import NumberCountsFactory
from firecrown.likelihood._cmb import CMBConvergenceFactory
from firecrown.likelihood._likelihood import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools
from firecrown.metadata_types import (
    Galaxies,
    TwoPointHarmonic,
    TwoPointReal,
    TypeSource,
    CMB,
    Clusters,
    GALAXY_SOURCE_TYPES,
    GALAXY_LENS_TYPES,
    CMB_TYPES,
)
from firecrown.data_types import TwoPointMeasurement
from firecrown.data_functions import (
    TwoPointBinFilterCollection,
    TwoPointBinFilter,
    extract_all_harmonic_data,
)
from firecrown.utils import (
    base_model_from_yaml,
    base_model_to_yaml,
    ClIntegrationOptions,
    ClIntegrationMethod,
    ClLimberMethod,
)
from firecrown.likelihood.factories._models import _build_two_point_likelihood_real


@pytest.fixture(name="empty_factory_harmonic")
def fixture_empty_factory_harmonic() -> TwoPointFactory:
    """Return an empty TwoPointFactory object."""
    return TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        weak_lensing_factories=[
            WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
        ],
        number_counts_factories=[
            NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
        ],
        cmb_factories=[CMBConvergenceFactory()],
    )


@pytest.fixture(name="empty_factory_real")
def fixture_empty_factory_real() -> TwoPointFactory:
    """Return an empty TwoPointFactory object."""
    return TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.REAL,
        weak_lensing_factories=[
            WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
        ],
        number_counts_factories=[
            NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
        ],
    )


def test_two_point_factory_dict() -> None:
    two_point_factory_dict = {
        "correlation_space": TwoPointCorrelationSpace.HARMONIC,
        "weak_lensing_factories": [
            {"per_bin_systematics": [], "global_systematics": []}
        ],
        "number_counts_factories": [
            {"per_bin_systematics": [], "global_systematics": []}
        ],
    }
    two_point_factory = TwoPointFactory.model_validate(two_point_factory_dict)
    assert isinstance(two_point_factory, TwoPointFactory)
    assert isinstance(two_point_factory.weak_lensing_factories[0], WeakLensingFactory)
    assert isinstance(two_point_factory.number_counts_factories[0], NumberCountsFactory)
    assert two_point_factory.correlation_space == TwoPointCorrelationSpace.HARMONIC
    assert two_point_factory.weak_lensing_factories[0].per_bin_systematics == []
    assert two_point_factory.weak_lensing_factories[0].global_systematics == []
    assert two_point_factory.number_counts_factories[0].per_bin_systematics == []
    assert two_point_factory.number_counts_factories[0].global_systematics == []


@pytest.mark.parametrize(
    "correlation_space",
    [TwoPointCorrelationSpace.HARMONIC, TwoPointCorrelationSpace.REAL],
)
def test_two_point_factory_to_from_dict(correlation_space) -> None:
    two_point_factory = TwoPointFactory(
        correlation_space=correlation_space,
        weak_lensing_factories=[
            WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
        ],
        number_counts_factories=[
            NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
        ],
        int_options=ClIntegrationOptions(
            method=ClIntegrationMethod.LIMBER, limber_method=ClLimberMethod.GSL_QAG_QUAD
        ),
    )

    yaml_str = base_model_to_yaml(two_point_factory)
    two_point_factory_from_dict = base_model_from_yaml(TwoPointFactory, yaml_str)
    assert isinstance(two_point_factory_from_dict, TwoPointFactory)
    assert isinstance(
        two_point_factory_from_dict.weak_lensing_factories[0], WeakLensingFactory
    )
    assert isinstance(
        two_point_factory_from_dict.number_counts_factories[0], NumberCountsFactory
    )
    assert two_point_factory_from_dict.correlation_space == correlation_space
    assert (
        two_point_factory_from_dict.weak_lensing_factories[0].per_bin_systematics == []
    )
    assert (
        two_point_factory_from_dict.weak_lensing_factories[0].global_systematics == []
    )
    assert (
        two_point_factory_from_dict.number_counts_factories[0].per_bin_systematics == []
    )
    assert (
        two_point_factory_from_dict.number_counts_factories[0].global_systematics == []
    )

    assert two_point_factory_from_dict.int_options == two_point_factory.int_options


def test_two_point_factor_direct() -> None:
    two_point_factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        weak_lensing_factories=[
            WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
        ],
        number_counts_factories=[
            NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
        ],
    )

    assert two_point_factory.correlation_space == TwoPointCorrelationSpace.HARMONIC
    assert two_point_factory.weak_lensing_factories[0].per_bin_systematics == []
    assert two_point_factory.weak_lensing_factories[0].global_systematics == []
    assert two_point_factory.number_counts_factories[0].per_bin_systematics == []
    assert two_point_factory.number_counts_factories[0].global_systematics == []


def test_two_point_factor_invalid_correlation_space_type() -> None:
    two_point_factory_dict = {
        "correlation_space": 1.2,
        "weak_lensing_factories": [
            {"per_bin_systematics": [], "global_systematics": []}
        ],
        "number_counts_factories": [
            {"per_bin_systematics": [], "global_systematics": []}
        ],
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
        "weak_lensing_factories": [
            {"per_bin_systematics": [], "global_systematics": []}
        ],
        "number_counts_factories": [
            {"per_bin_systematics": [], "global_systematics": []}
        ],
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
    sacc_data = data_source_sacc.get_sacc_data()
    assert sacc_data is not None
    assert isinstance(sacc_data, sacc.Sacc)


def test_data_source_sacc_dict_absolute() -> None:
    sacc_file = Path("tests/bug_398.sacc.gz").absolute()
    data_source_sacc_dict = {"sacc_data_file": sacc_file.as_posix()}
    data_source_sacc = DataSourceSacc.model_validate(data_source_sacc_dict)
    assert isinstance(data_source_sacc, DataSourceSacc)
    assert data_source_sacc.sacc_data_file == sacc_file.as_posix()
    sacc_data = data_source_sacc.get_sacc_data()
    assert sacc_data is not None
    assert isinstance(sacc_data, sacc.Sacc)


def test_data_source_sacc_direct() -> None:
    data_source_sacc = DataSourceSacc(sacc_data_file="tests/bug_398.sacc.gz")
    assert data_source_sacc.sacc_data_file == "tests/bug_398.sacc.gz"
    sacc_data = data_source_sacc.get_sacc_data()
    assert sacc_data is not None
    assert isinstance(sacc_data, sacc.Sacc)


def test_data_source_sacc_direct_absolute() -> None:
    sacc_file = Path("tests/bug_398.sacc.gz").absolute()
    data_source_sacc = DataSourceSacc(sacc_data_file=sacc_file.as_posix())
    assert data_source_sacc.sacc_data_file == sacc_file.as_posix()
    sacc_data = data_source_sacc.get_sacc_data()
    assert sacc_data is not None
    assert isinstance(sacc_data, sacc.Sacc)


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


def test_data_source_sacc_get_filepath_throws() -> None:
    # absolute data file name, no path, no such file.
    dss = DataSourceSacc(sacc_data_file="/tmp/no such file.fits")
    with pytest.raises(
        FileNotFoundError, match="File /tmp/no such file.fits does not exist"
    ):
        _ = dss.get_filepath()
    # relative data file name, path present, no such file.
    dss = DataSourceSacc(sacc_data_file="no such file.fits")
    dss.set_path(Path("/tmp"))
    with pytest.raises(
        FileNotFoundError, match="File no such file.fits does not exist"
    ):
        _ = dss.get_filepath()


def test_two_point_experiment_dict() -> None:
    two_point_experiment_dict = {
        "two_point_factory": {
            "correlation_space": TwoPointCorrelationSpace.HARMONIC,
            "weak_lensing_factories": [
                {
                    "per_bin_systematics": [],
                    "global_systematics": [],
                }
            ],
            "number_counts_factories": [
                {
                    "per_bin_systematics": [],
                    "global_systematics": [],
                }
            ],
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
        two_point_experiment.two_point_factory.weak_lensing_factories[
            0
        ].per_bin_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.weak_lensing_factories[
            0
        ].global_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.number_counts_factories[
            0
        ].per_bin_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.number_counts_factories[
            0
        ].global_systematics
        == []
    )
    assert two_point_experiment.data_source.sacc_data_file == "tests/bug_398.sacc.gz"


def test_two_point_experiment_direct() -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=TwoPointFactory(
            correlation_space=TwoPointCorrelationSpace.HARMONIC,
            weak_lensing_factories=[
                WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
            ],
            number_counts_factories=[
                NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
            ],
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
        two_point_experiment.two_point_factory.weak_lensing_factories[
            0
        ].per_bin_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.weak_lensing_factories[
            0
        ].global_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.number_counts_factories[
            0
        ].per_bin_systematics
        == []
    )
    assert (
        two_point_experiment.two_point_factory.number_counts_factories[
            0
        ].global_systematics
        == []
    )
    assert two_point_experiment.data_source.sacc_data_file == "tests/bug_398.sacc.gz"


def relative_to_with_walk_up(source: Path, target: Path) -> Path:
    """Compute the relative path from source to target.

    This allows walk-up (parent directory) traversal. Equivalent to Path.relative_to
    with walk_up=True in Python 3.12.

    :param source: The reference path (starting point).
    :param target: The target path to compute the relative path to.

    :return: A Path object representing the relative path from source to target.
    """
    source = source.resolve()
    target = target.resolve()
    # Check if paths are on different drives (Windows)
    if source.drive != target.drive:
        raise ValueError("Paths are on different drives")
    source_parts = source.parts
    target_parts = target.parts
    # Find common prefix length
    common_len = 0
    for s, t in zip(source_parts, target_parts):
        if s != t:
            break
        common_len += 1
    # Number of parent directories to walk up
    walk_up_count = len(source_parts) - common_len
    # Relative path components
    relative_parts = [".."] * walk_up_count + list(target_parts[common_len:])
    if not relative_parts:
        return Path(".")
    return Path(*relative_parts)


@pytest.mark.slow
def test_build_two_point_likelihood_real(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    top_dir = request.config.rootpath
    absolute_sacc_path = top_dir / Path("tests/sacc_data.hdf5")
    sacc_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, absolute_sacc_path
    )

    tmp_experiment_file.write_text(f"""
two_point_factory:
  correlation_space: real
  weak_lensing_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  number_counts_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  int_options:
    method: limber
    limber_method: gsl_spline
data_source:
    sacc_data_file: {sacc_path_relative_to_tmp_path}
""")

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    likelihood, tools = build_two_point_likelihood(build_parameters)
    assert isinstance(likelihood, Likelihood)
    assert isinstance(tools, ModelingTools)


def test_build_two_point_likelihood_harmonic(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    top_dir = request.config.rootpath
    absolute_sacc_path = top_dir / Path("tests/bug_398.sacc.gz")
    sacc_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, absolute_sacc_path
    )

    tmp_experiment_file.write_text(f"""
two_point_factory:
  correlation_space: harmonic
  weak_lensing_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  number_counts_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  int_options:
    method: limber
    limber_method: gsl_spline
data_source:
    sacc_data_file: {sacc_path_relative_to_tmp_path}
""")

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    likelihood, tools = build_two_point_likelihood(build_parameters)
    assert isinstance(likelihood, Likelihood)
    assert isinstance(tools, ModelingTools)


def test_build_two_point_likelihood_real_no_real_data(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    top_dir = request.config.rootpath
    absolute_sacc_path = top_dir / Path("tests/bug_398.sacc.gz")
    sacc_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, absolute_sacc_path
    )

    tmp_experiment_file.write_text(f"""
two_point_factory:
  correlation_space: real
  weak_lensing_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  number_counts_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  int_options:
    method: limber
    limber_method: gsl_spline
data_source:
    sacc_data_file: {sacc_path_relative_to_tmp_path}
""")

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    with pytest.raises(
        ValueError,
        match=re.compile(
            "No two-point measurements in real space found in the SACC file.", re.DOTALL
        ),
    ):
        _ = build_two_point_likelihood(build_parameters)


def test_build_two_point_likelihood_harmonic_no_harmonic_data(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    top_dir = request.config.rootpath
    absolute_sacc_path = top_dir / Path("tests/sacc_data.hdf5")
    sacc_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, absolute_sacc_path
    )

    tmp_experiment_file.write_text(f"""
two_point_factory:
  correlation_space: harmonic
  weak_lensing_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  number_counts_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  int_options:
    method: limber
    limber_method: gsl_spline
data_source:
  sacc_data_file: {sacc_path_relative_to_tmp_path}
""")

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


def test_build_two_point_harmonic_with_filter(empty_factory_harmonic) -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_harmonic,
        data_source=DataSourceSacc(
            sacc_data_file="tests/bug_398.sacc.gz",
            filters=TwoPointBinFilterCollection(
                filters=[
                    TwoPointBinFilter.from_args_auto(
                        name=f"lens{i}",
                        measurement=Galaxies.COUNTS,
                        lower=2,
                        upper=3000,
                    )
                    for i in range(5)
                ],
                require_filter_for_all=False,
                allow_empty=False,
            ),
        ),
    )
    assert two_point_experiment.make_likelihood() is not None


def test_build_two_point_harmonic_with_filter_require_filter(
    empty_factory_harmonic,
) -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_harmonic,
        data_source=DataSourceSacc(
            sacc_data_file="tests/bug_398.sacc.gz",
            filters=TwoPointBinFilterCollection(
                filters=[
                    TwoPointBinFilter.from_args_auto(
                        name=f"lens{i}",
                        measurement=Galaxies.COUNTS,
                        lower=2,
                        upper=3000,
                    )
                    for i in range(5)
                ],
                require_filter_for_all=True,
                allow_empty=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="The bin name .* does not have a filter."):
        _ = two_point_experiment.make_likelihood()


def test_build_two_point_harmonic_with_filter_empty(empty_factory_harmonic) -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_harmonic,
        data_source=DataSourceSacc(
            sacc_data_file="tests/bug_398.sacc.gz",
            filters=TwoPointBinFilterCollection(
                filters=[
                    TwoPointBinFilter.from_args_auto(
                        name=f"lens{i}",
                        measurement=Galaxies.COUNTS,
                        lower=20000,
                        upper=30000,
                    )
                    for i in range(5)
                ],
                require_filter_for_all=False,
                allow_empty=False,
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match=(
            "The TwoPointMeasurement .* does "
            "not have any elements matching the filter."
        ),
    ):
        _ = two_point_experiment.make_likelihood()


@pytest.mark.slow
def test_build_two_point_real_with_filter(empty_factory_real: TwoPointFactory) -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_real,
        data_source=DataSourceSacc(
            sacc_data_file="tests/sacc_data.hdf5",
            filters=TwoPointBinFilterCollection(
                filters=[
                    TwoPointBinFilter.from_args_auto(
                        name=f"lens{i}",
                        measurement=Galaxies.COUNTS,
                        lower=0.0,
                        upper=100.0,
                    )
                    for i in range(5)
                ],
                require_filter_for_all=False,
                allow_empty=False,
            ),
        ),
    )
    assert two_point_experiment.make_likelihood() is not None


@pytest.mark.slow
def test_build_two_point_real_with_filter_allow_empty(
    empty_factory_real: TwoPointFactory,
) -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_real,
        data_source=DataSourceSacc(
            sacc_data_file="tests/sacc_data.hdf5",
            filters=TwoPointBinFilterCollection(
                filters=[
                    TwoPointBinFilter.from_args_auto(
                        name=f"lens{i}",
                        measurement=Galaxies.COUNTS,
                        lower=20000,
                        upper=30000,
                    )
                    for i in range(5)
                ],
                require_filter_for_all=False,
                allow_empty=True,
            ),
        ),
    )
    assert two_point_experiment.make_likelihood() is not None


def test_build_from_metadata_harmonic(
    empty_factory_harmonic: TwoPointFactory, two_point_cell: TwoPointHarmonic
) -> None:
    two_points = empty_factory_harmonic.from_metadata([two_point_cell])
    assert len(two_points) == 1
    two_point0 = two_points[0]
    assert isinstance(two_point0, TwoPoint)


def test_build_from_metadata_harmonic_cwindow(
    empty_factory_harmonic: TwoPointFactory,
    optimized_two_point_cwindow: TwoPointHarmonic,
) -> None:
    two_points = empty_factory_harmonic.from_metadata([optimized_two_point_cwindow])
    assert len(two_points) == 1
    two_point0 = two_points[0]
    assert isinstance(two_point0, TwoPoint)


def test_build_from_metadata_real(
    empty_factory_real: TwoPointFactory, optimized_two_point_real: TwoPointReal
) -> None:
    two_points = empty_factory_real.from_metadata([optimized_two_point_real])
    assert len(two_points) == 1
    two_point0 = two_points[0]
    assert isinstance(two_point0, TwoPoint)


def test_build_from_measurement_harmonic(
    empty_factory_harmonic: TwoPointFactory,
    harmonic_data_with_window: TwoPointMeasurement,
) -> None:
    two_points = empty_factory_harmonic.from_measurement([harmonic_data_with_window])
    assert len(two_points) == 1
    two_point0 = two_points[0]
    assert isinstance(two_point0, TwoPoint)


def test_build_from_measurement_harmonic_cwindow(
    empty_factory_harmonic: TwoPointFactory,
    harmonic_data_with_window: TwoPointMeasurement,
) -> None:
    two_points = empty_factory_harmonic.from_measurement([harmonic_data_with_window])
    assert len(two_points) == 1
    two_point0 = two_points[0]
    assert isinstance(two_point0, TwoPoint)


def test_two_point_factory_get_wl_factory():
    wl_factory1 = WeakLensingFactory(type_source=TypeSource("ts1"))
    wl_factory2 = WeakLensingFactory(type_source=TypeSource("ts2"))
    wl_factory_default = WeakLensingFactory()
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        weak_lensing_factories=[wl_factory1, wl_factory2, wl_factory_default],
    )
    for type_source, wl_factory in zip(
        [TypeSource("ts1"), TypeSource("ts2"), TypeSource.DEFAULT],
        [wl_factory1, wl_factory2, wl_factory_default],
    ):
        for measurement in GALAXY_SOURCE_TYPES:
            wl_factory0 = factory.get_factory(measurement, type_source=type_source)
            assert isinstance(wl_factory0, WeakLensingFactory)
            assert wl_factory0.type_source == type_source
            assert wl_factory0 is wl_factory


def test_two_point_factory_get_nc_factory():
    nc_factory1 = NumberCountsFactory(type_source=TypeSource("ts1"))
    nc_factory2 = NumberCountsFactory(type_source=TypeSource("ts2"))
    nc_factory_default = NumberCountsFactory()
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        number_counts_factories=[nc_factory1, nc_factory2, nc_factory_default],
    )
    for type_source, nc_factory in zip(
        [TypeSource("ts1"), TypeSource("ts2"), TypeSource.DEFAULT],
        [nc_factory1, nc_factory2, nc_factory_default],
    ):
        for measurement in GALAXY_LENS_TYPES:
            nc_factory0 = factory.get_factory(measurement, type_source=type_source)
            assert isinstance(nc_factory0, NumberCountsFactory)
            assert nc_factory0.type_source == type_source
            assert nc_factory0 is nc_factory


def test_two_point_factory_two_equal_wls():
    with pytest.raises(
        ValueError, match="Duplicate WeakLensingFactory found for type_source ts1"
    ):
        _ = TwoPointFactory(
            correlation_space=TwoPointCorrelationSpace.HARMONIC,
            weak_lensing_factories=[
                WeakLensingFactory(type_source=TypeSource("ts1")),
                WeakLensingFactory(type_source=TypeSource("ts1")),
            ],
        )


def test_two_point_factory_two_equal_ncs():
    with pytest.raises(
        ValueError, match="Duplicate NumberCountsFactory found for type_source ts1"
    ):
        _ = TwoPointFactory(
            correlation_space=TwoPointCorrelationSpace.HARMONIC,
            number_counts_factories=[
                NumberCountsFactory(type_source=TypeSource("ts1")),
                NumberCountsFactory(type_source=TypeSource("ts1")),
            ],
        )


def test_two_point_factory_get_unavailable_wl_factory():
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        weak_lensing_factories=[
            WeakLensingFactory(type_source=TypeSource("ts1")),
        ],
    )
    with pytest.raises(
        ValueError, match="No WeakLensingFactory found for type_source ts2."
    ):
        _ = factory.get_factory(Galaxies.SHEAR_E, type_source=TypeSource("ts2"))


def test_two_point_factory_get_unavailable_nc_factory():
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        number_counts_factories=[
            NumberCountsFactory(type_source=TypeSource("ts1")),
        ],
    )
    with pytest.raises(
        ValueError, match="No NumberCountsFactory found for type_source ts2."
    ):
        _ = factory.get_factory(Galaxies.COUNTS, type_source=TypeSource("ts2"))


@pytest.mark.parametrize(
    "file, expected",
    [
        # Test with string input
        ("example.txt", Path("example.txt")),
        # Test with Path object
        (Path("example.txt"), Path("example.txt")),
        # Test with absolute path string
        ("/home/user/example.txt", Path("/home/user/example.txt")),
        # Test with relative path string
        ("../example.txt", Path("../example.txt")),
    ],
)
def test_ensure_path(file, expected):
    result = ensure_path(file)
    assert result == expected


def test_ensure_path_invalid():
    with pytest.raises(
        AssertionError, match="Expected code to be unreachable, but got: 123"
    ):
        ensure_path(123)  # type: ignore


def test_build_two_point_likelihood_harmonic_filters_result_empty(
    empty_factory_harmonic,
) -> None:
    """Test case where filters remove all data in harmonic space."""
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_harmonic,
        data_source=DataSourceSacc(
            sacc_data_file="tests/bug_398.sacc.gz",
            filters=TwoPointBinFilterCollection(
                filters=[
                    # Filter that won't match any data
                    TwoPointBinFilter.from_args_auto(
                        name="nonexistent_lens",
                        measurement=Galaxies.COUNTS,
                        lower=2,
                        upper=3000,
                    )
                ],
                require_filter_for_all=False,
                allow_empty=True,  # Allow empty after filtering
            ),
        ),
    )
    # This should succeed because allow_empty=True
    likelihood = two_point_experiment.make_likelihood()
    assert likelihood is not None


def test_two_point_experiment_with_ccl_factory() -> None:
    """Test TwoPointExperiment with custom CCLFactory."""
    from firecrown.modeling_tools import CCLFactory

    custom_ccl_factory = CCLFactory()
    two_point_experiment = TwoPointExperiment(
        two_point_factory=TwoPointFactory(
            correlation_space=TwoPointCorrelationSpace.HARMONIC,
            weak_lensing_factories=[
                WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
            ],
            number_counts_factories=[
                NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
            ],
        ),
        data_source=DataSourceSacc(sacc_data_file="tests/bug_398.sacc.gz"),
        ccl_factory=custom_ccl_factory,
    )
    # The ccl_factory should remain the one we provided
    assert two_point_experiment.ccl_factory is custom_ccl_factory


def test_data_source_sacc_with_path_file_exists() -> None:
    """Test DataSourceSacc.get_filepath when path is set and file exists."""
    data_source = DataSourceSacc(sacc_data_file="bug_398.sacc.gz")
    data_source.set_path(Path("tests"))
    filepath = data_source.get_filepath()
    assert filepath == Path("tests/bug_398.sacc.gz")


def test_data_source_sacc_absolute_path_exists() -> None:
    """Test DataSourceSacc.get_filepath with absolute path that exists."""
    absolute_path = Path("tests/bug_398.sacc.gz").absolute()
    data_source = DataSourceSacc(sacc_data_file=str(absolute_path))
    filepath = data_source.get_filepath()
    assert filepath == absolute_path


def test_build_two_point_likelihood_real_fast() -> None:
    """Fast test for real-space likelihood building using direct function call."""
    # Create a minimal SACC file with real-space data
    sacc_data = sacc.Sacc()

    # Add minimal tracers
    sacc_data.add_tracer("NZ", "lens0", z=np.linspace(0.1, 1.0, 10), nz=np.ones(10))

    # Add minimal real-space data points
    theta = np.array([1.0, 2.0, 3.0])  # angular separation in arcmin
    xi = np.array([0.01, 0.005, 0.002])  # correlation function values
    error = np.array([0.001, 0.001, 0.001])  # errors

    for t, x, e in zip(theta, xi, error):
        sacc_data.add_data_point(
            "galaxy_density_xi", ("lens0", "lens0"), x, error=e, theta=t
        )

    # Set a minimal covariance matrix
    sacc_data.add_covariance(np.diag(error**2))

    # Create a minimal factory
    two_point_factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.REAL,
        weak_lensing_factories=[],
        number_counts_factories=[
            NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
        ],
    )

    # This should exercise lines 218-225 in _build_two_point_likelihood_real
    likelihood = _build_two_point_likelihood_real(sacc_data, two_point_factory)
    assert likelihood is not None


def test_build_two_point_likelihood_real_with_filters_fast() -> None:
    """Fast test for real-space likelihood building with filters."""
    # Create a minimal SACC file with real-space data
    sacc_data = sacc.Sacc()

    # Add minimal tracers
    sacc_data.add_tracer("NZ", "lens0", z=np.linspace(0.1, 1.0, 10), nz=np.ones(10))

    # Add minimal real-space data points
    theta = np.array([1.0, 2.0, 3.0])
    xi = np.array([0.01, 0.005, 0.002])
    error = np.array([0.001, 0.001, 0.001])

    for t, x, e in zip(theta, xi, error):
        sacc_data.add_data_point(
            "galaxy_density_xi", ("lens0", "lens0"), x, error=e, theta=t
        )

    # Set a minimal covariance matrix
    sacc_data.add_covariance(np.diag(error**2))

    # Create a minimal factory
    two_point_factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.REAL,
        weak_lensing_factories=[],
        number_counts_factories=[
            NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
        ],
    )

    # Create filters that won't remove data (allow_empty=True)
    filters = TwoPointBinFilterCollection(
        filters=[
            TwoPointBinFilter.from_args_auto(
                name="lens0",
                measurement=Galaxies.COUNTS,
                lower=0.5,  # Should include our theta values
                upper=5.0,
            )
        ],
        require_filter_for_all=False,
        allow_empty=True,
    )

    # This should exercise the filter branch in lines 219-220
    likelihood = _build_two_point_likelihood_real(
        sacc_data, two_point_factory, filters=filters
    )
    assert likelihood is not None


def test_two_point_factory_two_equal_cmbs():
    """Test that duplicate CMBConvergenceFactory with same type_source raises error."""
    with pytest.raises(
        ValueError, match="Duplicate CMBConvergenceFactory found for type_source ts1"
    ):
        _ = TwoPointFactory(
            correlation_space=TwoPointCorrelationSpace.HARMONIC,
            cmb_factories=[
                CMBConvergenceFactory(type_source=TypeSource("ts1")),
                CMBConvergenceFactory(type_source=TypeSource("ts1")),
            ],
        )


def test_two_point_factory_get_cmb_factory():
    """Test getting CMB factories by type_source."""
    cmb_factory1 = CMBConvergenceFactory(type_source=TypeSource("ts1"))
    cmb_factory2 = CMBConvergenceFactory(type_source=TypeSource("ts2"))
    cmb_factory_default = CMBConvergenceFactory()
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        cmb_factories=[cmb_factory1, cmb_factory2, cmb_factory_default],
    )

    for type_source, cmb_factory in zip(
        [TypeSource("ts1"), TypeSource("ts2"), TypeSource.DEFAULT],
        [cmb_factory1, cmb_factory2, cmb_factory_default],
    ):
        for measurement in CMB_TYPES:  # You'll need to import this
            cmb_factory0 = factory.get_factory(measurement, type_source=type_source)
            assert isinstance(cmb_factory0, CMBConvergenceFactory)
            assert cmb_factory0.type_source == type_source
            assert cmb_factory0 is cmb_factory


def test_two_point_factory_get_unavailable_cmb_factory():
    """Test error when requesting unavailable CMB factory type_source."""
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        cmb_factories=[
            CMBConvergenceFactory(type_source=TypeSource("ts1")),
        ],
    )
    with pytest.raises(
        ValueError, match="No CMBConvergenceFactory found for type_source ts2."
    ):
        _ = factory.get_factory(CMB.CONVERGENCE, type_source=TypeSource("ts2"))


def test_two_point_factory_cmb_measurement_handling():
    """Test that CMB measurements are properly handled by get_factory."""
    cmb_factory = CMBConvergenceFactory()
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        cmb_factories=[cmb_factory],
    )

    # Test that CMB.CONVERGENCE measurement returns CMB factory
    result_factory = factory.get_factory(CMB.CONVERGENCE)
    assert isinstance(result_factory, CMBConvergenceFactory)
    assert result_factory is cmb_factory


def test_two_point_factory_cmb_measurement_with_type_source():
    """Test CMB measurement handling with specific type_source."""
    cmb_factory1 = CMBConvergenceFactory(type_source=TypeSource("ts1"))
    cmb_factory2 = CMBConvergenceFactory(type_source=TypeSource("ts2"))

    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        cmb_factories=[cmb_factory1, cmb_factory2],
    )

    # Test that specific type_source returns correct factory
    result_factory1 = factory.get_factory(
        CMB.CONVERGENCE, type_source=TypeSource("ts1")
    )
    assert isinstance(result_factory1, CMBConvergenceFactory)
    assert result_factory1 is cmb_factory1

    result_factory2 = factory.get_factory(
        CMB.CONVERGENCE, type_source=TypeSource("ts2")
    )
    assert isinstance(result_factory2, CMBConvergenceFactory)
    assert result_factory2 is cmb_factory2


def test_two_point_factory_all_cmb_types():
    """Test that all CMB measurement types are handled correctly."""
    cmb_factory = CMBConvergenceFactory()
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        cmb_factories=[cmb_factory],
    )

    # Test all CMB measurement types
    for measurement in CMB_TYPES:
        result_factory = factory.get_factory(measurement)
        assert isinstance(result_factory, CMBConvergenceFactory)
        assert result_factory is cmb_factory


def test_two_point_factory_measurement_not_supported():
    """Test that measurements not supported by the factory raise an error."""
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        cmb_factories=[CMBConvergenceFactory()],
    )
    with pytest.raises(
        ValueError,
        match=("Factory not found for measurement .*, it is not supported."),
    ):
        _ = factory.get_factory(Clusters.COUNTS)


def test_two_point_factory_all_factories():
    """Test that all factory types are handled correctly."""
    wl_factory = WeakLensingFactory(type_source=TypeSource("ts_wl"))
    nc_factory = NumberCountsFactory(type_source=TypeSource("ts_nc"))
    cmb_factory = CMBConvergenceFactory(type_source=TypeSource("ts_cmb"))
    factory = TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        weak_lensing_factories=[wl_factory],
        number_counts_factories=[nc_factory],
        cmb_factories=[cmb_factory],
    )
    # Test Weak Lensing
    wl_result = factory.get_factory(Galaxies.SHEAR_E, type_source=TypeSource("ts_wl"))
    assert isinstance(wl_result, WeakLensingFactory)
    assert wl_result is wl_factory
    # Test Number Counts
    nc_result = factory.get_factory(Galaxies.COUNTS, type_source=TypeSource("ts_nc"))
    assert isinstance(nc_result, NumberCountsFactory)
    assert nc_result is nc_factory
    # Test CMB
    cmb_result = factory.get_factory(CMB.CONVERGENCE, type_source=TypeSource("ts_cmb"))
    assert isinstance(cmb_result, CMBConvergenceFactory)
    assert cmb_result is cmb_factory


def test_load_sacc_data_file_not_found():
    """Test load_sacc_data raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError, match="SACC file not found"):
        load_sacc_data("nonexistent_file.fits")


def test_load_sacc_data_file_not_found_with_path():
    """Test load_sacc_data raises FileNotFoundError with Path object."""
    with pytest.raises(FileNotFoundError, match="SACC file not found"):
        load_sacc_data(Path("/tmp/nonexistent_file.sacc"))


def test_load_sacc_data_both_formats_fail(tmp_path):
    """Test load_sacc_data raises ValueError when file is neither HDF5 nor FITS."""
    # Create a file with invalid content (not HDF5, not FITS)
    bad_file = tmp_path / "corrupted.sacc"
    bad_file.write_text("This is not a valid SACC file format")

    with pytest.raises(
        ValueError,
        match=re.compile(
            "Failed to load SACC data from file.*"
            "The file could not be read as either HDF5 or FITS format",
            re.DOTALL,
        ),
    ):
        load_sacc_data(bad_file)


def test_load_sacc_data_with_path_object():
    """Test load_sacc_data works with Path object input."""
    sacc_data = load_sacc_data(Path("tests/bug_398.sacc.gz"))
    assert sacc_data is not None
    assert isinstance(sacc_data, sacc.Sacc)


def test_load_sacc_data_with_string():
    """Test load_sacc_data works with string input."""
    sacc_data = load_sacc_data("tests/bug_398.sacc.gz")
    assert sacc_data is not None
    assert isinstance(sacc_data, sacc.Sacc)


@pytest.fixture(name="sacc_with_window", scope="function")
def fixture_sacc_with_window(tmp_path: Path):
    """Create a SACC file with window functions for testing normalize_window."""
    # Create window function data
    n_ell_bin = 5
    ell_bin_edges = np.geomspace(10, 500, n_ell_bin + 1)
    ell_bin_centers = np.sqrt(ell_bin_edges[:-1] * ell_bin_edges[1:])
    ell_bin_widths = np.diff(ell_bin_edges)

    ell_window = np.arange(800)
    window_function = np.zeros((ell_window.shape[0], n_ell_bin))
    for i, (mu_ell, sigma_ell) in enumerate(zip(ell_bin_centers, ell_bin_widths)):
        window_function[:, i] = np.exp(
            -0.5 * (ell_window - mu_ell) ** 2 / (sigma_ell) ** 2
        )

    # Intentionally NOT normalizing - so we can test the difference
    # window_function = window_function / window_function.sum(axis=0, keepdims=True)

    # Create SACC data
    sacc_data = sacc.Sacc()
    z = np.linspace(0, 2.0, 50)

    # Add tracers
    for i in range(2):
        dndz = np.exp(-0.5 * (z - 0.5 * (i + 1)) ** 2 / 0.1**2)
        sacc_data.add_tracer("NZ", f"src{i}", z, dndz)

    # Create cosmology for realistic data
    cosmo = pyccl.CosmologyVanillaLCDM()
    ccl_tracers = {}
    for i in range(2):
        tracer = sacc_data.get_tracer(f"src{i}")
        ccl_tracers[f"src{i}"] = pyccl.WeakLensingTracer(
            cosmo=cosmo, dndz=(tracer.z, tracer.nz)
        )

    # Add data with window functions
    window = sacc.BandpowerWindow(ell_window, window_function)
    for i in range(2):
        for j in range(i, 2):
            Cell = pyccl.angular_cl(
                cosmo=cosmo,
                tracer1=ccl_tracers[f"src{i}"],
                tracer2=ccl_tracers[f"src{j}"],
                ell=ell_window,
            )
            Cell_binned = window_function.T @ Cell
            sacc_data.add_ell_cl(
                data_type="galaxy_shear_cl_ee",
                tracer1=f"src{i}",
                tracer2=f"src{j}",
                ell=ell_bin_centers,
                x=Cell_binned,
                window=window,
            )

    # Add covariance
    sacc_data.add_covariance(np.identity(len(sacc_data)) * 0.01)

    # Save to file
    sacc_file = tmp_path / "test_window.fits"
    sacc_data.save_fits(str(sacc_file), overwrite=True)

    return sacc_file


def test_data_source_sacc_normalize_window_default(sacc_with_window: Path):
    """Test that DataSourceSacc.normalize_window defaults to True."""
    data_source = DataSourceSacc(sacc_data_file=str(sacc_with_window))
    assert data_source.normalize_window is True


def test_data_source_sacc_normalize_window_false_dict(sacc_with_window: Path):
    """Test DataSourceSacc.normalize_window can be set to False via dict."""
    data_source_dict = {
        "sacc_data_file": str(sacc_with_window),
        "normalize_window": False,
    }
    data_source = DataSourceSacc.model_validate(data_source_dict)
    assert data_source.normalize_window is False


def test_data_source_sacc_normalize_window_false_direct(sacc_with_window: Path):
    """Test DataSourceSacc.normalize_window can be set to False directly."""
    data_source = DataSourceSacc(
        sacc_data_file=str(sacc_with_window), normalize_window=False
    )
    assert data_source.normalize_window is False


def test_data_source_sacc_normalize_window_true_explicit(sacc_with_window: Path):
    """Test DataSourceSacc.normalize_window can be explicitly set to True."""
    data_source = DataSourceSacc(
        sacc_data_file=str(sacc_with_window), normalize_window=True
    )
    assert data_source.normalize_window is True


def test_two_point_experiment_normalize_window_flows_through(
    empty_factory_harmonic: TwoPointFactory,
    sacc_with_window: Path,
):
    """Test that normalize_window flows through TwoPointExperiment to likelihood."""
    # Create experiment with normalize_window=False
    experiment = TwoPointExperiment(
        two_point_factory=empty_factory_harmonic,
        data_source=DataSourceSacc(
            sacc_data_file=str(sacc_with_window), normalize_window=False
        ),
    )

    # Verify the attribute is set correctly
    assert experiment.data_source.normalize_window is False

    # Make likelihood - this should not raise an error
    likelihood = experiment.make_likelihood()
    assert likelihood is not None


def test_two_point_experiment_normalize_window_default_flows_through(
    empty_factory_harmonic: TwoPointFactory,
    sacc_with_window: Path,
):
    """Test that default normalize_window=True flows through correctly."""
    experiment = TwoPointExperiment(
        two_point_factory=empty_factory_harmonic,
        data_source=DataSourceSacc(sacc_data_file=str(sacc_with_window)),
    )

    # Verify the default is True
    assert experiment.data_source.normalize_window is True

    # Make likelihood - this should not raise an error
    likelihood = experiment.make_likelihood()
    assert likelihood is not None


def test_normalize_window_affects_extraction(sacc_with_window: Path):
    """Test that normalize_window parameter actually affects window extraction."""
    sacc_data = load_sacc_data(sacc_with_window)

    # Extract with normalization (default)
    data_normalized = extract_all_harmonic_data(sacc_data, normalize=True)

    # Extract without normalization
    data_unnormalized = extract_all_harmonic_data(sacc_data, normalize=False)

    # Both should have the same number of measurements
    assert len(data_normalized) == len(data_unnormalized)

    # Check if any measurements have window functions
    has_window = False
    for measurement_norm, measurement_unnorm in zip(data_normalized, data_unnormalized):
        assert isinstance(measurement_norm.metadata, TwoPointHarmonic)
        assert isinstance(measurement_unnorm.metadata, TwoPointHarmonic)
        if measurement_norm.metadata.window is not None:
            has_window = True
            window_norm = measurement_norm.metadata.window
            window_unnorm = measurement_unnorm.metadata.window

            # Windows should exist in both
            assert window_unnorm is not None

            # Normalized window should sum to 1 along axis 0
            assert np.allclose(window_norm.sum(axis=0), 1.0)

            # Unnormalized window should NOT necessarily sum to 1
            # (unless it happens to by coincidence)
            # But we can verify that normalized != unnormalized (when they differ)
            if not np.allclose(window_unnorm.sum(axis=0), 1.0):
                # If unnormalized doesn't sum to 1, they should be different
                assert not np.allclose(window_norm, window_unnorm)

                # The normalized should be the unnormalized divided by its sum
                expected_normalized = window_unnorm / window_unnorm.sum(axis=0)
                assert np.allclose(window_norm, expected_normalized)

    # Verify we actually tested something with windows
    assert has_window, "No window functions found in test data"


def test_two_point_experiment_yaml_with_normalize_window(
    tmp_path: Path, sacc_with_window: Path
):
    """Test TwoPointExperiment loading from YAML with normalize_window=False."""
    tmp_experiment_file = tmp_path / "experiment_no_norm.yaml"
    # Use relative path from tmp_path to sacc_with_window
    sacc_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, sacc_with_window
    )

    tmp_experiment_file.write_text(f"""
two_point_factory:
  correlation_space: harmonic
  weak_lensing_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
  number_counts_factories:
    - type_source: default
      per_bin_systematics: []
      global_systematics: []
data_source:
  sacc_data_file: {sacc_path_relative_to_tmp_path}
  normalize_window: false
""")

    # Load the experiment
    experiment = TwoPointExperiment.load_from_yaml(tmp_experiment_file)

    # Verify normalize_window is set to False
    assert experiment.data_source.normalize_window is False

    # Create likelihood to ensure it works end-to-end
    likelihood = experiment.make_likelihood()
    assert likelihood is not None
