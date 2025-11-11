"""
Tests for the module firecrown.metadata_types and firecrown.metadata_functions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from firecrown.metadata_types import (
    TwoPointHarmonic,
    TwoPointXY,
    TwoPointReal,
    _type_to_sacc_string_harmonic as harmonic,
    _type_to_sacc_string_real as real,
)
from firecrown.data_types import TwoPointMeasurement


def test_two_point_cells_with_data(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.array(np.ones_like(ells) * 1.1, dtype=np.float64)
    indices = np.arange(100)
    covariance_name = "cov"
    tpm = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy),
    )
    metadata = tpm.metadata
    assert isinstance(metadata, TwoPointHarmonic)

    assert metadata.ells[0] == 0
    assert metadata.ells[-1] == 100
    assert tpm.data is not None
    assert_array_equal(tpm.data, data)
    assert_array_equal(tpm.indices, indices)
    assert tpm.covariance_name == covariance_name


def test_two_point_two_point_cwindow_with_data(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.zeros((100, 4), dtype=np.float64)
    # Create a window with 4 bins, each containing 25 elements with a weight of 1.0.
    # The bins are defined as follows:
    # - Bin 1: Elements 0 to 24
    # - Bin 2: Elements 25 to 49
    # - Bin 3: Elements 50 to 74
    # - Bin 4: Elements 75 to 99
    rows = np.arange(100)
    cols = rows // 25
    weights[rows, cols] = 1.0
    window_ells = np.array([0, 1, 2, 3], dtype=np.float64)

    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.array(np.zeros(4) + 1.1, dtype=np.float64)
    indices = np.arange(4)
    covariance_name = "cov"
    tpm = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointHarmonic(
            ells=ells, window=weights, window_ells=window_ells, XY=harmonic_two_point_xy
        ),
    )
    metadata = tpm.metadata
    assert isinstance(metadata, TwoPointHarmonic)

    assert metadata.window is not None
    assert_array_equal(metadata.window, weights)
    assert metadata.XY == harmonic_two_point_xy
    assert metadata.get_sacc_name() == harmonic(
        harmonic_two_point_xy.x_measurement, harmonic_two_point_xy.y_measurement
    )
    assert tpm.data is not None
    assert_array_equal(tpm.data, data)
    assert_array_equal(tpm.indices, indices)
    assert tpm.covariance_name == covariance_name


def test_two_point_xi_theta_with_data(optimized_real_two_point_xy: TwoPointXY):
    thetas = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    data = np.array(np.zeros(100) + 1.1, dtype=np.float64)
    indices = np.arange(100)
    covariance_name = "cov"
    tpm = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    metadata = tpm.metadata
    assert isinstance(metadata, TwoPointReal)

    assert metadata.XY == optimized_real_two_point_xy
    assert metadata.get_sacc_name() == real(
        optimized_real_two_point_xy.x_measurement,
        optimized_real_two_point_xy.y_measurement,
    )
    assert_array_equal(tpm.data, data)
    assert_array_equal(tpm.indices, indices)
    assert tpm.covariance_name == covariance_name


def test_two_point_cells_with_invalid_data_size(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.array(np.zeros(101) + 1.1, dtype=np.float64)
    indices = np.arange(101)
    covariance_name = "cov"

    with pytest.raises(
        ValueError,
        match="Data and metadata should have the same length.",
    ):
        TwoPointMeasurement(
            data=data,
            indices=indices,
            covariance_name=covariance_name,
            metadata=TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy),
        )


def test_two_point_cwindow_with_invalid_data_size(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)
    window_ells = np.array([0, 1, 2, 3], dtype=np.float64)

    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.array(np.zeros(5) + 1.1, dtype=np.float64)
    indices = np.arange(5)
    covariance_name = "cov"

    with pytest.raises(
        ValueError,
        match="Data and metadata should have the same length.",
    ):
        TwoPointMeasurement(
            data=data,
            indices=indices,
            covariance_name=covariance_name,
            metadata=TwoPointHarmonic(
                XY=harmonic_two_point_xy,
                ells=ells,
                window=weights,
                window_ells=window_ells,
            ),
        )


def test_two_point_xi_theta_with_invalid_data_size(
    optimized_real_two_point_xy: TwoPointXY,
):
    thetas = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    data = np.array(np.zeros(101) + 1.1, dtype=np.float64)
    indices = np.arange(101)
    covariance_name = "cov"

    with pytest.raises(
        ValueError,
        match="Data and metadata should have the same length.",
    ):
        TwoPointMeasurement(
            data=data,
            indices=indices,
            covariance_name=covariance_name,
            metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
        )


def test_two_point_measurement_invalid_data(optimized_real_two_point_xy: TwoPointXY):
    thetas = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    data = np.array([1, 2, 3, 4, 5], dtype=np.float64).reshape(-1, 1)
    indices = np.array([1, 2, 3, 4, 5])
    covariance_name = "cov"
    with pytest.raises(
        ValueError,
        match="Data should be a 1D array.",
    ):
        TwoPointMeasurement(
            data=data,
            indices=indices,
            covariance_name=covariance_name,
            metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
        )


def test_two_point_measurement_invalid_indices(optimized_real_two_point_xy: TwoPointXY):
    thetas = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    covariance_name = "cov"
    with pytest.raises(
        ValueError,
        match="Data and indices should have the same shape.",
    ):
        TwoPointMeasurement(
            data=data,
            indices=indices,
            covariance_name=covariance_name,
            metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
        )


def test_two_point_measurement_eq(optimized_real_two_point_xy: TwoPointXY):
    thetas = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([1, 2, 3, 4, 5])
    covariance_name = "cov"
    measure_1 = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    measure_2 = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    assert measure_1 == measure_2


def test_two_point_measurement_neq(optimized_real_two_point_xy: TwoPointXY):
    thetas = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    indices = np.array([1, 2, 3, 4, 5])
    covariance_name = "cov"
    measure_1 = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    measure_2 = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name="cov2",
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    assert measure_1 != measure_2
    measure_3 = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    measure_4 = TwoPointMeasurement(
        data=data,
        indices=indices + 1,
        covariance_name=covariance_name,
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    assert measure_3 != measure_4
    measure_5 = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    measure_6 = TwoPointMeasurement(
        data=np.array(data + 1.0, dtype=np.float64),
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas),
    )
    assert measure_5 != measure_6
