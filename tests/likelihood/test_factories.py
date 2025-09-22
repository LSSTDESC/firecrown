"""Tests for the module firecrown.likelihood.factories."""

import re
from pathlib import Path
import pytest

import sacc
from firecrown.likelihood.factories import (
    build_two_point_likelihood,
    DataSourceSacc,
    ensure_path,
    TwoPointCorrelationSpace,
    TwoPointExperiment,
    TwoPointFactory,
)
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.weak_lensing import WeakLensingFactory
from firecrown.likelihood.number_counts import NumberCountsFactory
from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools
from firecrown.metadata_types import (
    Galaxies,
    TwoPointHarmonic,
    TwoPointReal,
    TypeSource,
    GALAXY_SOURCE_TYPES,
    GALAXY_LENS_TYPES,
)
from firecrown.data_types import TwoPointMeasurement
from firecrown.data_functions import TwoPointBinFilterCollection, TwoPointBinFilter
from firecrown.utils import (
    base_model_from_yaml,
    base_model_to_yaml,
    ClIntegrationOptions,
    ClIntegrationMethod,
    ClLimberMethod,
)


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


def test_build_two_point_likelihood_real(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    top_dir = request.config.rootpath
    absolute_fits_path = top_dir / Path("examples/des_y1_3x2pt/sacc_data.fits")
    fits_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, absolute_fits_path
    )

    tmp_experiment_file.write_text(
        f"""
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
    sacc_data_file: {fits_path_relative_to_tmp_path}
"""
    )

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    likelihood, tools = build_two_point_likelihood(build_parameters)
    assert isinstance(likelihood, Likelihood)
    assert isinstance(tools, ModelingTools)


def test_build_two_point_likelihood_harmonic(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    top_dir = request.config.rootpath
    absolute_fits_path = top_dir / Path("tests/bug_398.sacc.gz")
    fits_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, absolute_fits_path
    )

    tmp_experiment_file.write_text(
        f"""
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
    sacc_data_file: {fits_path_relative_to_tmp_path}
"""
    )

    build_parameters = NamedParameters({"likelihood_config": str(tmp_experiment_file)})
    likelihood, tools = build_two_point_likelihood(build_parameters)
    assert isinstance(likelihood, Likelihood)
    assert isinstance(tools, ModelingTools)


def test_build_two_point_likelihood_real_no_real_data(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    top_dir = request.config.rootpath
    absolute_fits_path = top_dir / Path("tests/bug_398.sacc.gz")
    fits_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, absolute_fits_path
    )

    tmp_experiment_file.write_text(
        f"""
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
    sacc_data_file: {fits_path_relative_to_tmp_path}
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


def test_build_two_point_likelihood_harmonic_no_harmonic_data(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    tmp_experiment_file = tmp_path / "experiment.yaml"
    top_dir = request.config.rootpath
    absolute_fits_path = top_dir / Path("examples/des_y1_3x2pt/sacc_data.fits")
    fits_path_relative_to_tmp_path = relative_to_with_walk_up(
        tmp_path, absolute_fits_path
    )

    tmp_experiment_file.write_text(
        f"""
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
  sacc_data_file: {fits_path_relative_to_tmp_path}
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
            sacc_data_file="examples/des_y1_3x2pt/sacc_data.fits",
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
def test_build_two_point_real_with_filter_require_filter(
    empty_factory_real: TwoPointFactory,
) -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_real,
        data_source=DataSourceSacc(
            sacc_data_file="examples/des_y1_3x2pt/sacc_data.fits",
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
                require_filter_for_all=True,
                allow_empty=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="The bin name .* does not have a filter."):
        _ = two_point_experiment.make_likelihood()


@pytest.mark.slow
def test_build_two_point_real_with_filter_empty(
    empty_factory_real: TwoPointFactory,
) -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_real,
        data_source=DataSourceSacc(
            sacc_data_file="examples/des_y1_3x2pt/sacc_data.fits",
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
def test_build_two_point_real_with_filter_allow_empty(
    empty_factory_real: TwoPointFactory,
) -> None:
    two_point_experiment = TwoPointExperiment(
        two_point_factory=empty_factory_real,
        data_source=DataSourceSacc(
            sacc_data_file="examples/des_y1_3x2pt/sacc_data.fits",
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
    empty_factory_harmonic: TwoPointFactory, two_point_cwindow: TwoPointHarmonic
) -> None:
    two_points = empty_factory_harmonic.from_metadata([two_point_cwindow])
    assert len(two_points) == 1
    two_point0 = two_points[0]
    assert isinstance(two_point0, TwoPoint)


def test_build_from_metadata_real(
    empty_factory_real: TwoPointFactory, two_point_real: TwoPointReal
) -> None:
    two_points = empty_factory_real.from_metadata([two_point_real])
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
