"""
Tests for the module firecrown.data_functions.
"""

from typing import Literal
import pytest
import numpy as np

from firecrown.metadata_types import (
    TwoPointReal,
    TwoPointHarmonic,
    InferredGalaxyZDist,
)
from firecrown.metadata_functions import make_all_photoz_bin_combinations
from firecrown.data_types import TwoPointMeasurement
from firecrown.data_functions import TwoPointBinFilter, TwoPointBinFilterCollection


@pytest.fixture(name="filter_merge_mode", params=["intersection", "union"])
def fixture_filter_merge_mode(request) -> str:
    """Fixture for the filter merge mode."""
    return request.param


@pytest.fixture(name="harmonic_bins")
def fixture_harmonic_bins(
    harmonic_bin_1: InferredGalaxyZDist, harmonic_bin_2: InferredGalaxyZDist
) -> list[TwoPointMeasurement]:
    """Create a list of TwoPointMeasurement with harmonic metadata."""
    all_xy = make_all_photoz_bin_combinations([harmonic_bin_1, harmonic_bin_2])
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
    harmonic_bin_1: InferredGalaxyZDist, harmonic_bin_2: InferredGalaxyZDist
) -> list[TwoPointMeasurement]:
    """Create a list of TwoPointMeasurement with harmonic metadata."""
    all_xy = make_all_photoz_bin_combinations([harmonic_bin_1, harmonic_bin_2])
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
    real_bin_1: InferredGalaxyZDist, real_bin_2: InferredGalaxyZDist
) -> list[TwoPointMeasurement]:
    """Create a list of TwoPointMeasurement with real metadata."""
    all_xy = make_all_photoz_bin_combinations([real_bin_1, real_bin_2])
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


@pytest.fixture(
    name="harmonic_filter_collection",
    params=[
        [("bin_1", (5, 60))],
        [("bin_2", (40, 95))],
        [("bin_1", (5, 60)), ("bin_2", (40, 95))],
    ],
)
def fixture_harmonic_filter_collection(
    request, filter_merge_mode: Literal["intersection", "union"]
) -> TwoPointBinFilterCollection:
    """Create a TwoPointBinFilterCollection with harmonic filters."""
    bin_filters = [
        TwoPointBinFilter(bin_name=bin_name, bin_filter=bin_filter)
        for bin_name, bin_filter in request.param
    ]
    return TwoPointBinFilterCollection(
        bin_filters=bin_filters, filter_merge_mode=filter_merge_mode
    )


@pytest.fixture(
    name="real_filter_collection",
    params=[
        [("bin_1", (0.1, 0.6))],
        [("bin_2", (0.4, 0.9))],
        [("bin_1", (0.1, 0.6)), ("bin_2", (0.4, 0.9))],
    ],
)
def fixture_real_filter_collection(
    request, filter_merge_mode: Literal["intersection", "union"]
) -> TwoPointBinFilterCollection:
    """Create a TwoPointBinFilterCollection with real filters."""
    bin_filters = [
        TwoPointBinFilter(bin_name=bin_name, bin_filter=bin_filter)
        for bin_name, bin_filter in request.param
    ]
    return TwoPointBinFilterCollection(
        bin_filters=bin_filters, filter_merge_mode=filter_merge_mode
    )


def test_two_point_bin_filter_construct():
    bin_filter = TwoPointBinFilter(bin_name="bin_1", bin_filter=(0.1, 0.5))
    assert bin_filter.bin_name == "bin_1"
    assert bin_filter.bin_filter == (0.1, 0.5)


def test_two_point_bin_filter_construct_invalid_range():
    with pytest.raises(
        ValueError, match="Value error, The bin filter should be a valid range."
    ):
        TwoPointBinFilter(bin_name="bin_2", bin_filter=(0.5, 0.1))


def test_two_point_bin_filter_collection_construct(filter_merge_mode):
    bin_filter_1 = TwoPointBinFilter(bin_name="bin_1", bin_filter=(0.1, 0.5))
    bin_filter_2 = TwoPointBinFilter(bin_name="bin_2", bin_filter=(0.5, 0.9))
    bin_filter_collection = TwoPointBinFilterCollection(
        bin_filters=[bin_filter_1, bin_filter_2],
        filter_merge_mode=filter_merge_mode,
    )
    assert bin_filter_collection.bin_filters == [bin_filter_1, bin_filter_2]
    assert bin_filter_collection.bin_filter_dict == {
        "bin_1": (0.1, 0.5),
        "bin_2": (0.5, 0.9),
    }


def test_two_point_bin_filter_collection_run_merge_filter(
    filter_merge_mode: Literal["intersection", "union"]
) -> None:
    bin_filter_1 = TwoPointBinFilter(bin_name="bin_1", bin_filter=(0.1, 0.5))
    bin_filter_2 = TwoPointBinFilter(bin_name="bin_2", bin_filter=(0.5, 0.9))
    bin_filter_collection = TwoPointBinFilterCollection(
        bin_filters=[bin_filter_1, bin_filter_2],
        filter_merge_mode=filter_merge_mode,
    )

    vals = np.linspace(0, 1, 100, dtype=np.float64)
    match_elements = bin_filter_collection.run_merge_filter(
        (0.3, 0.9), (0.1, 0.6), vals
    )
    assert len(match_elements) == 100
    if filter_merge_mode == "intersection":
        assert np.all(match_elements == (vals >= 0.3) & (vals <= 0.6))
    elif filter_merge_mode == "union":
        assert np.all(match_elements == (vals >= 0.1) & (vals <= 0.9))


def test_two_point_bin_filter_collection_construct_same_name(
    filter_merge_mode: Literal["intersection", "union"],
) -> None:
    bin_filter_1 = TwoPointBinFilter(bin_name="bin_1", bin_filter=(0.1, 0.5))
    bin_filter_2 = TwoPointBinFilter(bin_name="bin_1", bin_filter=(0.5, 0.9))
    with pytest.raises(
        ValueError, match="The bin name bin_1 is repeated in the bin filters."
    ):
        TwoPointBinFilterCollection(
            bin_filters=[bin_filter_1, bin_filter_2],
            filter_merge_mode=filter_merge_mode,
        )


def test_two_point_harmonic_bin_filter_collection_filter_match(
    harmonic_filter_collection: TwoPointBinFilterCollection,
    harmonic_bins: list[TwoPointMeasurement],
) -> None:
    for harmonic_bin in harmonic_bins:
        name_x = harmonic_bin.metadata.XY.x.bin_name
        name_y = harmonic_bin.metadata.XY.y.bin_name

        if harmonic_filter_collection.filter_match(harmonic_bin):
            assert name_x in harmonic_filter_collection.bin_filter_dict
            assert name_y in harmonic_filter_collection.bin_filter_dict
        else:
            assert (name_x not in harmonic_filter_collection.bin_filter_dict) or (
                name_y not in harmonic_filter_collection.bin_filter_dict
            )


def test_two_point_real_bin_filter_collection_filter_match(
    real_filter_collection: TwoPointBinFilterCollection,
    real_bins: list[TwoPointMeasurement],
) -> None:
    for harmonic_bin in real_bins:
        name_x = harmonic_bin.metadata.XY.x.bin_name
        name_y = harmonic_bin.metadata.XY.y.bin_name

        if real_filter_collection.filter_match(harmonic_bin):
            assert name_x in real_filter_collection.bin_filter_dict
            assert name_y in real_filter_collection.bin_filter_dict
        else:
            assert (name_x not in real_filter_collection.bin_filter_dict) or (
                name_y not in real_filter_collection.bin_filter_dict
            )


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
        match_ells = harmonic_filter_collection.run_merge_filter(
            harmonic_filter_collection.bin_filter_dict[
                harmonic_bin.metadata.XY.x.bin_name
            ],
            harmonic_filter_collection.bin_filter_dict[
                harmonic_bin.metadata.XY.y.bin_name
            ],
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
        match_ells = harmonic_filter_collection.run_merge_filter(
            harmonic_filter_collection.bin_filter_dict[
                harmonic_bin.metadata.XY.x.bin_name
            ],
            harmonic_filter_collection.bin_filter_dict[
                harmonic_bin.metadata.XY.y.bin_name
            ],
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
        assert isinstance(harmonic_bin.metadata, TwoPointReal)
        assert np.all(
            match_elements
            == real_filter_collection.run_merge_filter(
                real_filter_collection.bin_filter_dict[
                    harmonic_bin.metadata.XY.x.bin_name
                ],
                real_filter_collection.bin_filter_dict[
                    harmonic_bin.metadata.XY.y.bin_name
                ],
                harmonic_bin.metadata.thetas,
            )
        )


def test_two_point_harmonic_bin_filter_collection_call(
    harmonic_filter_collection: TwoPointBinFilterCollection,
    harmonic_bins: list[TwoPointMeasurement],
) -> None:
    filtered_bins = harmonic_filter_collection(harmonic_bins)
    assert len(filtered_bins) <= len(harmonic_bins)
    tracer_names_dict = {
        bin.metadata.XY.get_tracer_names(): bin for bin in harmonic_bins
    }
    filtered_tracer_names_dict = {
        bin.metadata.XY.get_tracer_names(): bin for bin in filtered_bins
    }
    # All filtered tracer names should be in the original tracer names list
    assert all(
        filtered_tracer_names in tracer_names_dict
        for filtered_tracer_names in filtered_tracer_names_dict
    )

    for filtered_tracer_names, filtered_bin in filtered_tracer_names_dict.items():
        original_bin = tracer_names_dict[filtered_tracer_names]
        assert isinstance(filtered_bin.metadata, TwoPointHarmonic)
        assert isinstance(original_bin.metadata, TwoPointHarmonic)
        match_elements = harmonic_filter_collection.run_merge_filter(
            harmonic_filter_collection.bin_filter_dict[
                original_bin.metadata.XY.x.bin_name
            ],
            harmonic_filter_collection.bin_filter_dict[
                original_bin.metadata.XY.y.bin_name
            ],
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
    tracer_names_dict = {
        bin.metadata.XY.get_tracer_names(): bin for bin in harmonic_window_bins
    }
    filtered_tracer_names_dict = {
        bin.metadata.XY.get_tracer_names(): bin for bin in filtered_bins
    }
    # All filtered tracer names should be in the original tracer names list
    assert all(
        filtered_tracer_names in tracer_names_dict
        for filtered_tracer_names in filtered_tracer_names_dict
    )

    for filtered_tracer_names, filtered_bin in filtered_tracer_names_dict.items():
        original_bin = tracer_names_dict[filtered_tracer_names]
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


def test_two_point_real_bin_filter_collection_call(
    real_filter_collection: TwoPointBinFilterCollection,
    real_bins: list[TwoPointMeasurement],
) -> None:
    filtered_bins = real_filter_collection(real_bins)
    assert len(filtered_bins) <= len(real_bins)
    tracer_names_dict = {bin.metadata.XY.get_tracer_names(): bin for bin in real_bins}
    filtered_tracer_names_dict = {
        bin.metadata.XY.get_tracer_names(): bin for bin in filtered_bins
    }
    # All filtered tracer names should be in the original tracer names list
    assert all(
        filtered_tracer_names in tracer_names_dict
        for filtered_tracer_names in filtered_tracer_names_dict
    )

    for filtered_tracer_names, filtered_bin in filtered_tracer_names_dict.items():
        original_bin = tracer_names_dict[filtered_tracer_names]
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
