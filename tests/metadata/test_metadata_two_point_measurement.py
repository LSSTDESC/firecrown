"""
Tests for the module firecrown.metadata_types and firecrown.metadata_functions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from firecrown.metadata_types import (
    TwoPointHarmonic,
    TwoPointMeasurement,
    TwoPointXY,
    TwoPointReal,
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
)


def test_two_point_cells_with_data(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.ones_like(ells) * 1.1
    indices = np.arange(100)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )

    cells = TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy, Cell=measure)

    assert cells.ells[0] == 0
    assert cells.ells[-1] == 100
    assert cells.Cell is not None
    assert cells.has_data()
    assert_array_equal(cells.Cell.data, data)
    assert_array_equal(cells.Cell.indices, indices)
    assert cells.Cell.covariance_name == covariance_name


def test_two_point_two_point_cwindow_with_data(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.zeros(4) + 1.1
    indices = np.arange(4)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )

    two_point = TwoPointHarmonic(
        XY=harmonic_two_point_xy, ells=ells, window=weights, Cell=measure
    )

    assert two_point.window is not None
    assert_array_equal(two_point.window, weights)
    assert two_point.XY == harmonic_two_point_xy
    assert two_point.get_sacc_name() == harmonic(
        harmonic_two_point_xy.x_measurement, harmonic_two_point_xy.y_measurement
    )
    assert two_point.has_data()
    assert two_point.Cell is not None
    assert_array_equal(two_point.Cell.data, data)
    assert_array_equal(two_point.Cell.indices, indices)
    assert two_point.Cell.covariance_name == covariance_name


def test_two_point_xi_theta_with_data(real_two_point_xy: TwoPointXY):
    data = np.zeros(100) + 1.1
    indices = np.arange(100)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )
    thetas = np.linspace(0.0, 1.0, 100)

    xi_theta = TwoPointReal(XY=real_two_point_xy, thetas=thetas, xis=measure)

    assert xi_theta.XY == real_two_point_xy
    assert xi_theta.get_sacc_name() == real(
        real_two_point_xy.x_measurement, real_two_point_xy.y_measurement
    )
    assert xi_theta.has_data()
    assert xi_theta.xis is not None
    assert_array_equal(xi_theta.xis.data, data)
    assert_array_equal(xi_theta.xis.indices, indices)
    assert xi_theta.xis.covariance_name == covariance_name


def test_two_point_cells_with_invalid_data_size(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.zeros(101) + 1.1
    indices = np.arange(101)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )

    with pytest.raises(
        ValueError,
        match=(
            "Data should have the same number of elements as the "
            "number of observations."
        ),
    ):
        TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy, Cell=measure)


def test_two_point_cwindow_with_invalid_data_size(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.zeros(5) + 1.1
    indices = np.arange(5)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )

    with pytest.raises(
        ValueError,
        match=(
            "Data should have the same number of elements as the number "
            "of observations."
        ),
    ):
        TwoPointHarmonic(
            XY=harmonic_two_point_xy, ells=ells, window=weights, Cell=measure
        )


def test_two_point_xi_theta_with_invalid_data_size(real_two_point_xy: TwoPointXY):
    data = np.zeros(101) + 1.1
    indices = np.arange(101)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )
    thetas = np.linspace(0.0, 1.0, 100)

    with pytest.raises(
        ValueError,
        match="Xis should have the same shape as thetas.",
    ):
        TwoPointReal(XY=real_two_point_xy, thetas=thetas, xis=measure)


def test_two_point_measurement_invalid_data():
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    indices = np.array([1, 2, 3, 4, 5])
    covariance_name = "cov"
    with pytest.raises(
        ValueError,
        match="Data should be a 1D array.",
    ):
        TwoPointMeasurement(data=data, indices=indices, covariance_name=covariance_name)


def test_two_point_measurement_invalid_indices():
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    covariance_name = "cov"
    with pytest.raises(
        ValueError,
        match="Data and indices should have the same shape.",
    ):
        TwoPointMeasurement(data=data, indices=indices, covariance_name=covariance_name)


def test_two_point_measurement_eq():
    data = np.array([1, 2, 3, 4, 5])
    indices = np.array([1, 2, 3, 4, 5])
    covariance_name = "cov"
    measure_1 = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )
    measure_2 = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )
    assert measure_1 == measure_2


def test_two_point_measurement_neq():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = np.array([1, 2, 3, 4, 5])
    covariance_name = "cov"
    measure_1 = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
    )
    measure_2 = TwoPointMeasurement(data=data, indices=indices, covariance_name="cov2")
    assert measure_1 != measure_2
    measure_3 = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )
    measure_4 = TwoPointMeasurement(
        data=data, indices=indices + 1, covariance_name=covariance_name
    )
    assert measure_3 != measure_4
    measure_5 = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )
    measure_6 = TwoPointMeasurement(
        data=data + 1.0, indices=indices, covariance_name=covariance_name
    )
    assert measure_5 != measure_6
