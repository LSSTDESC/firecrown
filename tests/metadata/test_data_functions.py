"""
Tests for the module firecrown.data_functions.
"""

import itertools as it
import pytest
import numpy as np

from firecrown.metadata_types import (
    TwoPointReal,
    TwoPointHarmonic,
    InferredGalaxyZDist,
    Galaxies,
)
from firecrown.metadata_functions import make_all_photoz_bin_combinations
from firecrown.data_types import TwoPointMeasurement
from firecrown.data_functions import (
    TwoPointBinFilter,
    TwoPointBinFilterCollection,
    TwoPointTracerSpec,
    bin_spec_from_metadata,
    make_interval_from_list,
)
from firecrown.utils import base_model_from_yaml, base_model_to_yaml


@pytest.fixture(name="harmonic_bins")
def fixture_harmonic_bins(
    all_harmonic_bins: list[InferredGalaxyZDist],
) -> list[TwoPointMeasurement]:
    """Create a list of TwoPointMeasurement with harmonic metadata."""
    all_xy = make_all_photoz_bin_combinations(all_harmonic_bins)
    data = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    indices = np.arange(100, dtype=np.int64)
    ells = np.arange(2, 102, dtype=np.int64)
    return [
        TwoPointMeasurement(
            metadata=TwoPointHarmonic(XY=xy, ells=ells, window=None),
            data=data,
            indices=indices,
            covariance_name="cov1",
        )
        for xy in all_xy
    ]


@pytest.fixture(name="harmonic_window_bins")
def fixture_harmonic_window_bins(
    all_harmonic_bins: list[InferredGalaxyZDist],
) -> list[TwoPointMeasurement]:
    """Create a list of TwoPointMeasurement with harmonic metadata."""
    all_xy = make_all_photoz_bin_combinations(all_harmonic_bins)
    data = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    indices = np.arange(10, dtype=np.int64)
    ells = np.arange(2, 102, dtype=np.int64)
    window = np.zeros((100, 10), dtype=np.float64)
    rows = np.arange(100)
    cols = rows // 10
    window[rows, cols] = 1.0

    return [
        TwoPointMeasurement(
            metadata=TwoPointHarmonic(XY=xy, ells=ells, window=window),
            data=data,
            indices=indices,
            covariance_name="cov1",
        )
        for xy in all_xy
    ]


@pytest.fixture(name="real_bins")
def fixture_real_bins(
    all_real_bins: list[InferredGalaxyZDist],
) -> list[TwoPointMeasurement]:
    """Create a list of TwoPointMeasurement with real metadata."""
    all_xy = make_all_photoz_bin_combinations(all_real_bins)
    data = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    indices = np.arange(100, dtype=np.int64)
    thetas = np.linspace(0.0, 0.25 * np.pi, 100, dtype=np.float64)
    return [
        TwoPointMeasurement(
            metadata=TwoPointReal(XY=xy, thetas=thetas),
            data=data,
            indices=indices,
            covariance_name="cov1",
        )
        for xy in all_xy
    ]


ALL_HARMONIC_BINS = list(
    it.combinations_with_replacement(
        tuple(it.product(("bin_1", "bin_2"), (Galaxies.COUNTS, Galaxies.SHEAR_E))),
        2,
    )
)


@pytest.fixture(
    name="harmonic_filter_collection",
    params=[
        elem
        for ncomb in range(len(ALL_HARMONIC_BINS) - 1, len(ALL_HARMONIC_BINS) + 1)
        for elem in it.combinations(ALL_HARMONIC_BINS, ncomb)
    ],
)
def fixture_harmonic_filter_collection(request) -> TwoPointBinFilterCollection:
    """Create a TwoPointBinFilterCollection with harmonic filters."""
    bin_filters = [
        TwoPointBinFilter.from_args(name_a, m_a, name_b, m_b, 20, 60)
        for (name_a, m_a), (name_b, m_b) in request.param
    ]

    return TwoPointBinFilterCollection(filters=bin_filters)


ALL_REAL_BINS = list(
    it.combinations_with_replacement(
        tuple(it.product(("bin_1", "bin_2"), [Galaxies.COUNTS, Galaxies.SHEAR_T])),
        2,
    )
)


@pytest.fixture(
    name="real_filter_collection",
    params=[
        elem
        for ncomb in range(len(ALL_REAL_BINS) - 1, len(ALL_REAL_BINS) + 1)
        for elem in it.combinations(ALL_REAL_BINS, ncomb)
    ],
)
def fixture_real_filter_collection(request) -> TwoPointBinFilterCollection:
    """Create a TwoPointBinFilterCollection with real filters."""
    bin_filters = [
        TwoPointBinFilter.from_args(name_a, m_a, name_b, m_b, 0.1, 0.5)
        for (name_a, m_a), (name_b, m_b) in request.param
    ]

    return TwoPointBinFilterCollection(filters=bin_filters)


def test_two_point_bin_filter_construct():
    bin_spec = [
        TwoPointTracerSpec(name="bin_1", measurement=Galaxies.COUNTS),
        TwoPointTracerSpec(name="bin_2", measurement=Galaxies.SHEAR_E),
    ]
    bin_filter = TwoPointBinFilter(spec=bin_spec, interval=(0.1, 0.5))
    assert bin_filter.spec == bin_spec
    assert bin_filter.interval == (0.1, 0.5)

    bin_filter_from_args = TwoPointBinFilter.from_args(
        "bin_1", Galaxies.COUNTS, "bin_2", Galaxies.SHEAR_E, 0.1, 0.5
    )

    assert bin_filter_from_args.spec == bin_spec
    assert bin_filter_from_args.interval == (0.1, 0.5)


def test_two_point_bin_filter_construct_auto():
    bin_spec = [TwoPointTracerSpec(name="bin_1", measurement=Galaxies.COUNTS)]
    bin_filter = TwoPointBinFilter(spec=bin_spec, interval=(0.1, 0.5))
    assert bin_filter.spec == bin_spec
    assert bin_filter.interval == (0.1, 0.5)

    bin_filter_from_args = TwoPointBinFilter.from_args_auto(
        "bin_1", Galaxies.COUNTS, 0.1, 0.5
    )

    assert bin_filter_from_args.spec == bin_spec
    assert bin_filter_from_args.interval == (0.1, 0.5)


def test_two_point_bin_filter_construct_invalid_range():
    bin_spec = [
        TwoPointTracerSpec(name="bin_1", measurement=Galaxies.COUNTS),
        TwoPointTracerSpec(name="bin_2", measurement=Galaxies.SHEAR_E),
    ]
    with pytest.raises(
        ValueError, match="Value error, The bin filter should be a valid range."
    ):
        TwoPointBinFilter(spec=bin_spec, interval=(0.5, 0.1))


def test_two_point_bin_filter_construct_empty_spec():
    with pytest.raises(
        ValueError, match="The bin_spec must contain one or two elements."
    ):
        TwoPointBinFilter(spec=[], interval=(0.1, 0.5))


def test_two_point_bin_filter_construct_too_many_spec():
    bin_spec = [
        TwoPointTracerSpec(name="bin_1", measurement=Galaxies.COUNTS),
        TwoPointTracerSpec(name="bin_2", measurement=Galaxies.SHEAR_E),
        TwoPointTracerSpec(name="bin_3", measurement=Galaxies.SHEAR_E),
    ]
    with pytest.raises(
        ValueError, match="The bin_spec must contain one or two elements."
    ):
        TwoPointBinFilter(spec=bin_spec, interval=(0.1, 0.5))


def test_two_point_bin_filter_collection_construct():
    bin_spec = (
        TwoPointTracerSpec(name="bin_1", measurement=Galaxies.COUNTS),
        TwoPointTracerSpec(name="bin_2", measurement=Galaxies.SHEAR_E),
    )
    bin_filter = TwoPointBinFilter.from_args(
        "bin_1", Galaxies.COUNTS, "bin_2", Galaxies.SHEAR_E, 0.1, 0.5
    )
    bin_filter_collection = TwoPointBinFilterCollection(filters=[bin_filter])
    assert bin_filter_collection.filters == [bin_filter]
    assert bin_filter_collection.bin_filter_dict == {frozenset(bin_spec): (0.1, 0.5)}


def test_two_point_bin_filter_collection_construct_same_name() -> None:
    bin_spec = [
        TwoPointTracerSpec(name="bin_1", measurement=Galaxies.COUNTS),
        TwoPointTracerSpec(name="bin_2", measurement=Galaxies.SHEAR_E),
    ]
    bin_filter_1 = TwoPointBinFilter(spec=bin_spec, interval=(0.1, 0.5))
    bin_filter_2 = TwoPointBinFilter(spec=bin_spec, interval=(0.5, 0.9))
    with pytest.raises(
        ValueError, match="The bin name .* is repeated in the bin filters."
    ):
        TwoPointBinFilterCollection(filters=[bin_filter_1, bin_filter_2])


def test_two_point_harmonic_bin_filter_collection_filter_match(
    harmonic_filter_collection: TwoPointBinFilterCollection,
    harmonic_bins: list[TwoPointMeasurement],
) -> None:
    for harmonic_bin in harmonic_bins:
        bin_spec = bin_spec_from_metadata(harmonic_bin.metadata)

        if harmonic_filter_collection.filter_match(harmonic_bin):
            assert bin_spec in harmonic_filter_collection.bin_filter_dict
        else:
            assert bin_spec not in harmonic_filter_collection.bin_filter_dict


def test_two_point_real_bin_filter_collection_filter_match(
    real_filter_collection: TwoPointBinFilterCollection,
    real_bins: list[TwoPointMeasurement],
) -> None:
    for harmonic_bin in real_bins:
        bin_spec = bin_spec_from_metadata(harmonic_bin.metadata)

        if real_filter_collection.filter_match(harmonic_bin):
            assert bin_spec in real_filter_collection.bin_filter_dict
        else:
            assert bin_spec not in real_filter_collection.bin_filter_dict


def test_two_point_harmonic_bin_filter_collection_apply_filter_single(
    harmonic_filter_collection: TwoPointBinFilterCollection,
    harmonic_bins: list[TwoPointMeasurement],
) -> None:
    for harmonic_bin in harmonic_bins:
        if not harmonic_filter_collection.filter_match(harmonic_bin):
            continue
        assert isinstance(harmonic_bin.metadata, TwoPointHarmonic)
        match_elements, match_obs = harmonic_filter_collection.apply_filter_single(
            harmonic_bin
        )
        bin_spec = bin_spec_from_metadata(harmonic_bin.metadata)
        match_ells = harmonic_filter_collection.run_bin_filter(
            harmonic_filter_collection.bin_filter_dict[bin_spec],
            harmonic_bin.metadata.ells,
        )
        assert np.all(match_elements == match_obs)
        assert np.all(match_elements == match_ells)


def test_two_point_harmonic_window_bin_filter_collection_apply_filter_single(
    harmonic_filter_collection: TwoPointBinFilterCollection,
    harmonic_window_bins: list[TwoPointMeasurement],
) -> None:
    for harmonic_bin in harmonic_window_bins:
        if not harmonic_filter_collection.filter_match(harmonic_bin):
            continue
        assert isinstance(harmonic_bin.metadata, TwoPointHarmonic)
        match_elements, match_obs = harmonic_filter_collection.apply_filter_single(
            harmonic_bin
        )
        bin_spec = bin_spec_from_metadata(harmonic_bin.metadata)
        match_ells = harmonic_filter_collection.run_bin_filter(
            harmonic_filter_collection.bin_filter_dict[bin_spec],
            harmonic_bin.metadata.ells,
        )
        assert np.all(match_elements == match_ells)
        # Ensure that every column excluded by match_elements has at least one non-zero
        # value in the rows excluded by match_ells within the harmonic_bin metadata
        # window.
        assert harmonic_bin.metadata.window is not None
        assert np.all(
            np.any(
                harmonic_bin.metadata.window[~match_elements, :][:, ~match_obs],
                axis=0,
            )
        )


def test_two_point_real_bin_filter_collection_apply_filter_single(
    real_filter_collection: TwoPointBinFilterCollection,
    real_bins: list[TwoPointMeasurement],
) -> None:
    for harmonic_bin in real_bins:
        if not real_filter_collection.filter_match(harmonic_bin):
            continue
        match_elements = real_filter_collection.apply_filter_single(harmonic_bin)
        bin_spec = bin_spec_from_metadata(harmonic_bin.metadata)
        assert isinstance(harmonic_bin.metadata, TwoPointReal)
        assert np.all(
            match_elements
            == real_filter_collection.run_bin_filter(
                real_filter_collection.bin_filter_dict[bin_spec],
                harmonic_bin.metadata.thetas,
            )
        )


def test_two_point_harmonic_bin_filter_collection_call(
    harmonic_filter_collection: TwoPointBinFilterCollection,
    harmonic_bins: list[TwoPointMeasurement],
) -> None:
    filtered_bins = harmonic_filter_collection(harmonic_bins)
    assert len(filtered_bins) <= len(harmonic_bins)
    bin_spec_dict = {bin_spec_from_metadata(bin.metadata): bin for bin in harmonic_bins}
    filtered_bin_spec_dict = {
        bin_spec_from_metadata(bin.metadata): bin for bin in filtered_bins
    }
    # All filtered bin_spec should be in the original bin_specs list
    assert all(bin_spec in bin_spec_dict for bin_spec in filtered_bin_spec_dict)

    for filtered_bin_spec, filtered_bin in filtered_bin_spec_dict.items():
        original_bin = bin_spec_dict[filtered_bin_spec]
        assert isinstance(filtered_bin.metadata, TwoPointHarmonic)
        assert isinstance(original_bin.metadata, TwoPointHarmonic)
        bin_spec = bin_spec_from_metadata(original_bin.metadata)
        match_elements = harmonic_filter_collection.run_bin_filter(
            harmonic_filter_collection.bin_filter_dict[bin_spec],
            original_bin.metadata.ells,
        )
        assert np.all(
            filtered_bin.metadata.ells == original_bin.metadata.ells[match_elements]
        )
        assert np.all(filtered_bin.data == original_bin.data[match_elements])
        assert np.all(filtered_bin.indices == original_bin.indices[match_elements])
        assert filtered_bin.covariance_name == original_bin.covariance_name
        assert filtered_bin.metadata.XY == original_bin.metadata.XY
        assert filtered_bin.metadata.window == original_bin.metadata.window


def test_two_point_harmonic_window_bin_filter_collection_call(
    harmonic_filter_collection: TwoPointBinFilterCollection,
    harmonic_window_bins: list[TwoPointMeasurement],
) -> None:
    filtered_bins = harmonic_filter_collection(harmonic_window_bins)
    assert len(filtered_bins) <= len(harmonic_window_bins)
    bin_specs_dict = {
        bin_spec_from_metadata(bin.metadata): bin for bin in harmonic_window_bins
    }
    filtered_bin_specs_dict = {
        bin_spec_from_metadata(bin.metadata): bin for bin in filtered_bins
    }
    # All filtered tracer names should be in the original tracer names list
    assert all(
        filtered_bin_specs in bin_specs_dict
        for filtered_bin_specs in filtered_bin_specs_dict
    )

    for filtered_bin_specs, filtered_bin in filtered_bin_specs_dict.items():
        if not harmonic_filter_collection.filter_match(filtered_bin):
            continue
        original_bin = bin_specs_dict[filtered_bin_specs]
        assert isinstance(filtered_bin.metadata, TwoPointHarmonic)
        assert isinstance(original_bin.metadata, TwoPointHarmonic)
        match_elements, match_obs = harmonic_filter_collection.apply_filter_single(
            original_bin
        )
        assert np.all(
            filtered_bin.metadata.ells == original_bin.metadata.ells[match_elements]
        )
        assert np.all(filtered_bin.data == original_bin.data[match_obs])
        assert np.all(filtered_bin.indices == original_bin.indices[match_obs])
        assert filtered_bin.covariance_name == original_bin.covariance_name
        assert filtered_bin.metadata.XY == original_bin.metadata.XY
        assert filtered_bin.metadata.window is not None
        assert original_bin.metadata.window is not None
        assert np.all(
            filtered_bin.metadata.window
            == original_bin.metadata.window[match_elements, :][:, match_obs]
        )


def test_two_point_harmonic_bin_filter_collection_call_require(
    harmonic_bin_1: InferredGalaxyZDist,
) -> None:
    harmonic_filter_collection_no_empty = TwoPointBinFilterCollection(
        filters=[
            TwoPointBinFilter.from_args(
                "bin_2", Galaxies.SHEAR_E, "bin_2", Galaxies.SHEAR_E, 5, 60
            )
        ],
        require_filter_for_all=True,
    )
    harmonic_bins = [
        TwoPointMeasurement(
            metadata=TwoPointHarmonic(
                XY=xy, ells=np.arange(2, 102, dtype=np.int64), window=None
            ),
            data=np.linspace(0.0, 1.0, 100, dtype=np.float64),
            indices=np.arange(100),
            covariance_name="cov1",
        )
        for xy in make_all_photoz_bin_combinations([harmonic_bin_1])
    ]
    with pytest.raises(ValueError, match="The bin name .* does not have a filter."):
        _ = harmonic_filter_collection_no_empty(harmonic_bins)


def test_two_point_harmonic_bin_filter_collection_call_no_empty(
    harmonic_bin_1: InferredGalaxyZDist,
) -> None:
    cm = list(harmonic_bin_1.measurements)[0]
    harmonic_filter_collection_no_empty = TwoPointBinFilterCollection(
        filters=[TwoPointBinFilter.from_args("bin_1", cm, "bin_1", cm, 1000, 2000)],
        require_filter_for_all=True,
    )
    harmonic_bins = [
        TwoPointMeasurement(
            metadata=TwoPointHarmonic(
                XY=xy, ells=np.arange(2, 102, dtype=np.int64), window=None
            ),
            data=np.linspace(0.0, 1.0, 100, dtype=np.float64),
            indices=np.arange(100),
            covariance_name="cov1",
        )
        for xy in make_all_photoz_bin_combinations([harmonic_bin_1])
    ]
    with pytest.raises(
        ValueError,
        match=(
            "The TwoPointMeasurement .* "
            "does not have any elements matching the filter."
        ),
    ):
        _ = harmonic_filter_collection_no_empty(harmonic_bins)


def test_two_point_harmonic_bin_filter_collection_call_empty(
    harmonic_bin_1: InferredGalaxyZDist,
) -> None:
    cm = list(harmonic_bin_1.measurements)[0]
    harmonic_filter_collection_no_empty = TwoPointBinFilterCollection(
        filters=[TwoPointBinFilter.from_args("bin_1", cm, "bin_1", cm, 1000, 2000)],
        allow_empty=True,
    )
    harmonic_bins = [
        TwoPointMeasurement(
            metadata=TwoPointHarmonic(
                XY=xy, ells=np.arange(2, 102, dtype=np.int64), window=None
            ),
            data=np.linspace(0.0, 1.0, 100, dtype=np.float64),
            indices=np.arange(100),
            covariance_name="cov1",
        )
        for xy in make_all_photoz_bin_combinations([harmonic_bin_1])
    ]
    filtered_harmonic_bins = harmonic_filter_collection_no_empty(harmonic_bins)
    assert len(filtered_harmonic_bins) == 0


def test_two_point_real_bin_filter_collection_call(
    real_filter_collection: TwoPointBinFilterCollection,
    real_bins: list[TwoPointMeasurement],
) -> None:
    filtered_bins = real_filter_collection(real_bins)
    assert len(filtered_bins) <= len(real_bins)
    bin_specs_dict = {bin_spec_from_metadata(bin.metadata): bin for bin in real_bins}
    filtered_bin_specs_dict = {
        bin_spec_from_metadata(bin.metadata): bin for bin in filtered_bins
    }
    # All filtered tracer names should be in the original tracer names list
    assert all(
        filtered_bin_specs in bin_specs_dict
        for filtered_bin_specs in filtered_bin_specs_dict
    )

    for filtered_bin_specs, filtered_bin in filtered_bin_specs_dict.items():
        original_bin = bin_specs_dict[filtered_bin_specs]
        assert isinstance(filtered_bin.metadata, TwoPointReal)
        assert isinstance(original_bin.metadata, TwoPointReal)
        match_elements, _ = real_filter_collection.apply_filter_single(original_bin)
        assert np.all(
            filtered_bin.metadata.thetas == original_bin.metadata.thetas[match_elements]
        )
        assert np.all(filtered_bin.data == original_bin.data[match_elements])
        assert np.all(filtered_bin.indices == original_bin.indices[match_elements])
        assert filtered_bin.covariance_name == original_bin.covariance_name
        assert filtered_bin.metadata.XY == original_bin.metadata.XY


def test_two_point_real_bin_filter_collection_call_require(
    real_bin_1: InferredGalaxyZDist,
) -> None:
    cm = list(real_bin_1.measurements)[0]
    real_filter_collection_no_empty = TwoPointBinFilterCollection(
        filters=[TwoPointBinFilter.from_args("bin_2", cm, "bin_2", cm, 0.1, 0.6)],
        require_filter_for_all=True,
    )
    real_bins = [
        TwoPointMeasurement(
            metadata=TwoPointReal(
                XY=xy, thetas=np.linspace(0.0, 0.25 * np.pi, 100, dtype=np.float64)
            ),
            data=np.linspace(0.0, 1.0, 100, dtype=np.float64),
            indices=np.arange(100),
            covariance_name="cov1",
        )
        for xy in make_all_photoz_bin_combinations([real_bin_1])
    ]
    with pytest.raises(ValueError, match="The bin name .* does not have a filter."):
        _ = real_filter_collection_no_empty(real_bins)


def test_two_point_real_bin_filter_collection_call_no_empty(
    real_bin_1: InferredGalaxyZDist,
) -> None:
    cm = list(real_bin_1.measurements)[0]
    real_filter_collection_no_empty = TwoPointBinFilterCollection(
        filters=[TwoPointBinFilter.from_args("bin_1", cm, "bin_1", cm, 10.1, 10.6)],
        require_filter_for_all=True,
    )
    real_bins = [
        TwoPointMeasurement(
            metadata=TwoPointReal(
                XY=xy, thetas=np.linspace(0.0, 0.25 * np.pi, 100, dtype=np.float64)
            ),
            data=np.linspace(0.0, 1.0, 100, dtype=np.float64),
            indices=np.arange(100),
            covariance_name="cov1",
        )
        for xy in make_all_photoz_bin_combinations([real_bin_1])
    ]
    with pytest.raises(
        ValueError,
        match=(
            "The TwoPointMeasurement .* "
            "does not have any elements matching the filter."
        ),
    ):
        _ = real_filter_collection_no_empty(real_bins)


def test_two_point_real_bin_filter_collection_call_empty(
    real_bin_1: InferredGalaxyZDist,
) -> None:
    cm = list(real_bin_1.measurements)[0]
    real_filter_collection_no_empty = TwoPointBinFilterCollection(
        filters=[TwoPointBinFilter.from_args("bin_1", cm, "bin_1", cm, 10.1, 10.6)],
        allow_empty=True,
    )
    real_bins = [
        TwoPointMeasurement(
            metadata=TwoPointReal(
                XY=xy, thetas=np.linspace(0.0, 0.25 * np.pi, 100, dtype=np.float64)
            ),
            data=np.linspace(0.0, 1.0, 100, dtype=np.float64),
            indices=np.arange(100),
            covariance_name="cov1",
        )
        for xy in make_all_photoz_bin_combinations([real_bin_1])
    ]
    filtered_real_bins = real_filter_collection_no_empty(real_bins)
    assert len(filtered_real_bins) == 0


def test_to_from_yaml_harmonic(
    harmonic_filter_collection: TwoPointBinFilterCollection,
) -> None:
    yaml = base_model_to_yaml(harmonic_filter_collection)
    assert (
        base_model_from_yaml(TwoPointBinFilterCollection, yaml)
        == harmonic_filter_collection
    )


def test_to_from_yaml_real(real_filter_collection: TwoPointBinFilterCollection) -> None:
    yaml = base_model_to_yaml(real_filter_collection)
    assert (
        base_model_from_yaml(TwoPointBinFilterCollection, yaml)
        == real_filter_collection
    )


def test_make_interval_from_list_tuple() -> None:
    interval = make_interval_from_list((0.1, 0.5))
    assert interval == (0.1, 0.5)


def test_make_interval_from_list_list() -> None:
    interval = make_interval_from_list([0.1, 0.5])
    assert interval == (0.1, 0.5)


def test_make_interval_from_list_list_wrong_len() -> None:
    with pytest.raises(ValueError, match="The list should have two values."):
        _ = make_interval_from_list([0.1, 0.5, 0.7])


def test_make_interval_from_list_list_wrong_element() -> None:
    with pytest.raises(ValueError, match="The list should have two float values."):
        _ = make_interval_from_list([0.1, "0.5"])  # type: ignore


def test_make_interval_from_list_wrong_type() -> None:
    with pytest.raises(ValueError, match="The values should be a list or a tuple."):
        _ = make_interval_from_list({0.1, 0.5})  # type: ignore
