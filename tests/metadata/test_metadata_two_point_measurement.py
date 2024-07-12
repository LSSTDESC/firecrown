"""
Tests for the module firecrown.metadata.two_point
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from firecrown.metadata.two_point import (
    Galaxies,
    InferredGalaxyZDist,
    TwoPointCells,
    TwoPointCWindow,
    TwoPointXiTheta,
    TwoPointXY,
    TwoPointMeasurement,
    Window,
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
)


def test_two_point_cells_with_data():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
    )
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.ones_like(ells) * 1.1
    indices = np.arange(100)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )

    cells = TwoPointCells(ells=ells, XY=xy, Cell=measure)

    assert cells.ells[0] == 0
    assert cells.ells[-1] == 100
    assert cells.Cell is not None
    assert cells.has_data()
    assert_array_equal(cells.Cell.data, data)
    assert_array_equal(cells.Cell.indices, indices)
    assert cells.Cell.covariance_name == covariance_name


def test_two_point_two_point_cwindow_with_data():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    ells_for_interpolation = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

    window = Window(
        ells=ells,
        weights=weights,
        ells_for_interpolation=ells_for_interpolation,
    )

    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
    )
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.zeros(4) + 1.1
    indices = np.arange(4)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )

    two_point = TwoPointCWindow(XY=xy, window=window, Cell=measure)

    assert two_point.window == window
    assert two_point.XY == xy
    assert two_point.get_sacc_name() == harmonic(xy.x_measurement, xy.y_measurement)
    assert two_point.has_data()
    assert two_point.Cell is not None
    assert_array_equal(two_point.Cell.data, data)
    assert_array_equal(two_point.Cell.indices, indices)
    assert two_point.Cell.covariance_name == covariance_name


def test_two_point_xi_theta_with_data():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
    )
    data = np.zeros(100) + 1.1
    indices = np.arange(100)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )
    thetas = np.linspace(0.0, 1.0, 100)

    xi_theta = TwoPointXiTheta(XY=xy, thetas=thetas, xis=measure)

    assert xi_theta.XY == xy
    assert xi_theta.get_sacc_name() == real(xy.x_measurement, xy.y_measurement)
    assert xi_theta.has_data()
    assert xi_theta.xis is not None
    assert_array_equal(xi_theta.xis.data, data)
    assert_array_equal(xi_theta.xis.indices, indices)
    assert xi_theta.xis.covariance_name == covariance_name


def test_two_point_cells_with_invalid_data():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
    )
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.zeros(101) + 1.1
    indices = np.arange(101)
    covariance_name = "cov"
    measure = TwoPointMeasurement(
        data=data, indices=indices, covariance_name=covariance_name
    )

    with pytest.raises(
        ValueError,
        match="Cell should have the same shape as ells.",
    ):
        TwoPointCells(ells=ells, XY=xy, Cell=measure)


def test_two_point_cwindow_with_invalid_data():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    ells_for_interpolation = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

    window = Window(
        ells=ells,
        weights=weights,
        ells_for_interpolation=ells_for_interpolation,
    )

    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
    )
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
            "of observations supported by the window function."
        ),
    ):
        TwoPointCWindow(XY=xy, window=window, Cell=measure)


def test_two_point_xi_theta_with_invalid_data():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
    )
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
        TwoPointXiTheta(XY=xy, thetas=thetas, xis=measure)


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
