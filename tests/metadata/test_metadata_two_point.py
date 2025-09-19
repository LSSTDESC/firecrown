"""
Tests for the module firecrown.metadata_types and firecrown.metadata_functions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from firecrown.metadata_types import (
    ALL_MEASUREMENTS,
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
    Clusters,
    CMB,
    Galaxies,
    InferredGalaxyZDist,
    TracerNames,
    TwoPointHarmonic,
    TwoPointXY,
    TwoPointReal,
)
from firecrown.data_types import TwoPointMeasurement
from firecrown.likelihood.source import SourceGalaxy
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.cmb import CMBConvergenceFactory


def test_inferred_galaxy_z_dist():
    z_dist = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    assert z_dist.bin_name == "bname1"
    assert z_dist.z[0] == 0
    assert z_dist.z[-1] == 1
    assert z_dist.dndz[0] == 1
    assert z_dist.dndz[-1] == 1
    assert z_dist.measurements == {Galaxies.COUNTS}


def test_inferred_galaxy_z_dist_bad_shape():
    with pytest.raises(
        ValueError, match="The z and dndz arrays should have the same shape."
    ):
        InferredGalaxyZDist(
            bin_name="bname1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(101),
            measurements={Clusters.COUNTS},
        )


def test_inferred_galaxy_z_dist_bad_type():
    with pytest.raises(ValueError, match="The measurement should be a Measurement."):
        InferredGalaxyZDist(
            bin_name="bname1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={0},  # type: ignore
        )


def test_inferred_galaxy_z_dist_bad_name():
    with pytest.raises(ValueError, match="The bin_name should not be empty."):
        InferredGalaxyZDist(
            bin_name="",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurements={Galaxies.COUNTS},
        )


def test_two_point_xy_gal_gal():
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
    assert xy.x == x
    assert xy.y == y
    assert xy.x_measurement == Galaxies.COUNTS
    assert xy.y_measurement == Galaxies.COUNTS


def test_two_point_xy_gal_gal_invalid_x_measurement():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    with pytest.raises(
        ValueError,
        match="Measurement .* not in the measurements of bname1.",
    ):
        TwoPointXY(
            x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
        )


def test_two_point_xy_gal_gal_invalid_y_measurement():
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
        measurements={Galaxies.SHEAR_E},
    )
    with pytest.raises(
        ValueError,
        match="Measurement .* not in the measurements of bname2.",
    ):
        TwoPointXY(
            x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
        )


def test_two_point_xy_cmb_gal():
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
        measurements={CMB.CONVERGENCE},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=CMB.CONVERGENCE
    )
    assert xy.x == x
    assert xy.y == y
    assert xy.x_measurement == Galaxies.COUNTS
    assert xy.y_measurement == CMB.CONVERGENCE


def test_two_point_xy_invalid():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    with pytest.raises(
        ValueError,
        match=("Measurements .* and .* are not compatible."),
    ):
        TwoPointXY(
            x=x, y=y, x_measurement=Galaxies.SHEAR_E, y_measurement=Galaxies.SHEAR_T
        )


def test_two_point_harmonic():
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
    cells = TwoPointHarmonic(ells=ells, XY=xy)

    assert_array_equal(cells.ells, ells)
    assert cells.XY == xy
    assert cells.get_sacc_name() == harmonic(xy.x_measurement, xy.y_measurement)
    assert cells.n_observations() == 100


def test_two_point_harmonic_invalid_ells():
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
    ells = np.array(np.linspace(0, 100), dtype=np.int64).reshape(-1, 10)
    with pytest.raises(
        ValueError,
        match="Ells should be a 1D array.",
    ):
        TwoPointHarmonic(ells=ells, XY=xy)


def test_two_point_harmonic_invalid_type():
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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    with pytest.raises(
        ValueError,
        match="Measurements .* and .* must support harmonic-space calculations.",
    ):
        TwoPointHarmonic(ells=ells, XY=xy)


def test_two_point_cwindow(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)
    window_ells = np.array([0, 1, 2, 3], dtype=np.float64)

    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    two_point = TwoPointHarmonic(
        XY=harmonic_two_point_xy, ells=ells, window=weights, window_ells=window_ells
    )

    assert two_point.window is not None
    assert_array_equal(two_point.window, weights)
    assert two_point.XY == harmonic_two_point_xy
    assert two_point.get_sacc_name() == harmonic(
        harmonic_two_point_xy.x_measurement, harmonic_two_point_xy.y_measurement
    )
    assert two_point.n_observations() == 4


def test_two_point_cwindow_wrong_data_shape(
    harmonic_two_point_xy: TwoPointXY,
):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)
    window_ells = np.array([0, 1, 2, 3], dtype=np.float64)

    data = (np.zeros(100) + 1.1).astype(np.float64)
    indices = np.arange(100)
    covariance_name = "cov"
    data = data.reshape(-1, 10)
    with pytest.raises(
        ValueError,
        match="Data should be a 1D array.",
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


def test_two_point_measurement_invalid_metadata():
    data = (np.zeros(100) + 1.1).astype(np.float64)
    indices = np.arange(100)
    covariance_name = "cov"
    with pytest.raises(
        ValueError,
        match="Metadata should be an instance of TwoPointReal or TwoPointHarmonic.",
    ):
        TwoPointMeasurement(
            data=data,
            indices=indices,
            covariance_name=covariance_name,
            metadata="Im not a metadata",  # type: ignore
        )


def test_two_point_cwindow_stringify(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)
    window_ells = np.array([0, 1, 2, 3], dtype=np.float64)

    two_point = TwoPointHarmonic(
        XY=harmonic_two_point_xy, ells=ells, window=weights, window_ells=window_ells
    )

    assert (
        str(two_point) == f"{str(harmonic_two_point_xy)}[{two_point.get_sacc_name()}]"
    )


def test_two_point_cwindow_invalid():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)
    window_ells = np.array([0, 1, 2, 3], dtype=np.float64)

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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    with pytest.raises(
        ValueError,
        match="Measurements .* and .* must support harmonic-space calculations.",
    ):
        TwoPointHarmonic(XY=xy, ells=ells, window=weights, window_ells=window_ells)


def test_two_point_cwindow_invalid_window():
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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )
    with pytest.raises(
        ValueError,
        match="window should be a ndarray.",
    ):
        TwoPointHarmonic(
            XY=xy, ells=np.array([1.0]), window="Im not a window"  # type: ignore
        )


def test_two_point_cwindow_invalid_window_shape():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400, dtype=np.float64)

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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )
    with pytest.raises(
        ValueError,
        match="window should be a 2D array.",
    ):
        TwoPointHarmonic(XY=xy, ells=ells, window=weights)


def test_two_point_cwindow_window_ell_not_match():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )
    with pytest.raises(
        ValueError,
        match="window should have the same number of rows as ells.",
    ):
        TwoPointHarmonic(XY=xy, ells=ells[:10], window=weights)


def test_two_point_cwindow_missing_window_ells():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )
    with pytest.raises(
        ValueError,
        match="window_ells must be set if window is set.",
    ):
        TwoPointHarmonic(XY=xy, ells=ells, window=weights)


def test_two_point_cwindow_window_ells_wrong_shape():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )

    with pytest.raises(
        ValueError,
        match="window_ells should be a 1D array.",
    ):
        TwoPointHarmonic(
            XY=xy,
            ells=ells,
            window=weights,
            window_ells=np.linspace(0, 1, 9).reshape(3, 3),
        )


def test_two_point_cwindow_window_ells_wrong_len():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )

    with pytest.raises(
        ValueError,
        match=(
            "window_ells should have the same number of "
            "elements as the columns of window."
        ),
    ):
        TwoPointHarmonic(
            XY=xy, ells=ells, window=weights, window_ells=np.linspace(0, 1, 9)
        )


def test_two_point_cwindow_no_window_with_window_ells():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)

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
        measurements={Galaxies.SHEAR_T},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_T
    )

    with pytest.raises(
        ValueError,
        match="window_ells must be None if window is None.",
    ):
        TwoPointHarmonic(XY=xy, ells=ells, window_ells=np.linspace(0, 1, 9))


def test_two_point_real():
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
    theta = np.array(np.linspace(0, 100, 100))
    two_point = TwoPointReal(XY=xy, thetas=theta)

    assert_array_equal(two_point.thetas, theta)
    assert two_point.XY == xy
    assert two_point.get_sacc_name() == real(xy.x_measurement, xy.y_measurement)
    assert two_point.n_observations() == 100


def test_two_point_real_invalid():
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
        measurements={Galaxies.SHEAR_E},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.SHEAR_E
    )
    theta = np.array(np.linspace(0, 100, 100))
    with pytest.raises(
        ValueError,
        match="Measurements .* and .* must support real-space calculations.",
    ):
        TwoPointReal(XY=xy, thetas=theta)


def test_harmonic_type_string_invalid():
    with pytest.raises(
        ValueError, match="Harmonic-space correlation not supported for shear T."
    ):
        harmonic(Galaxies.COUNTS, Galaxies.SHEAR_T)


def test_real_type_string_invalid():
    with pytest.raises(
        ValueError, match="Real-space correlation not supported for shear E."
    ):
        real(Galaxies.COUNTS, Galaxies.SHEAR_E)


def test_tracer_names_serialization():
    tn = TracerNames("x", "y")
    s = tn.to_yaml()
    recovered = TracerNames.from_yaml(s)
    assert tn == recovered


def test_measurement_serialization():
    for t in ALL_MEASUREMENTS:
        s = t.to_yaml()
        recovered = type(t).from_yaml(s)
        assert t == recovered


def test_inferred_galaxy_zdist_serialization(harmonic_bin_1: InferredGalaxyZDist):
    s = harmonic_bin_1.to_yaml()
    # Take a look at how hideous the generated string
    # is.
    recovered = InferredGalaxyZDist.from_yaml(s)
    assert harmonic_bin_1 == recovered


def test_two_point_xy_str(harmonic_two_point_xy: TwoPointXY):
    assert str(harmonic_two_point_xy) == (
        f"({harmonic_two_point_xy.x.bin_name}, " f"{harmonic_two_point_xy.y.bin_name})"
    )


def test_two_point_xy_serialization(harmonic_two_point_xy: TwoPointXY):
    s = harmonic_two_point_xy.to_yaml()
    # Take a look at how hideous the generated string
    # is.
    recovered = TwoPointXY.from_yaml(s)
    assert harmonic_two_point_xy == recovered
    assert str(harmonic_two_point_xy) == str(recovered)


def test_two_point_harmonic_str(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    cells = TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy)
    assert str(cells) == f"{str(harmonic_two_point_xy)}[{cells.get_sacc_name()}]"


def test_two_point_harmonic_serialization(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    cells = TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy)
    s = cells.to_yaml()
    recovered = TwoPointHarmonic.from_yaml(s)
    assert cells == recovered
    assert str(harmonic_two_point_xy) == str(recovered.XY)
    assert str(cells) == str(recovered)


def test_two_point_harmonic_cmp_invalid(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    cells = TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy)
    with pytest.raises(
        ValueError,
        match="Can only compare TwoPointHarmonic objects.",
    ):
        _ = cells == "Im not a TwoPointHarmonic"


def test_two_point_harmonic_ells_wrong_shape(harmonic_two_point_xy: TwoPointXY):
    ells = np.array(np.linspace(0, 100), dtype=np.int64).reshape(-1, 10)
    with pytest.raises(
        ValueError,
        match="Ells should be a 1D array.",
    ):
        TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy)


def test_two_point_harmonic_with_window_serialization(
    two_point_cwindow: TwoPointHarmonic,
):
    s = two_point_cwindow.to_yaml()
    recovered = TwoPointHarmonic.from_yaml(s)
    assert two_point_cwindow == recovered


def test_two_point_real_serialization(real_two_point_xy: TwoPointXY):
    theta = np.array(np.linspace(0, 10, 10))
    xi_theta = TwoPointReal(XY=real_two_point_xy, thetas=theta)
    s = xi_theta.to_yaml()
    recovered = TwoPointReal.from_yaml(s)
    assert xi_theta == recovered
    assert str(real_two_point_xy) == str(recovered.XY)
    assert str(xi_theta) == str(recovered)


def test_two_point_real_cmp_invalid(real_two_point_xy: TwoPointXY):
    theta = np.array(np.linspace(0, 10, 10))
    xi_theta = TwoPointReal(XY=real_two_point_xy, thetas=theta)
    with pytest.raises(
        ValueError,
        match="Can only compare TwoPointReal objects.",
    ):
        _ = xi_theta == "Im not a TwoPointReal"


def test_two_point_real_wrong_shape(real_two_point_xy: TwoPointXY):
    theta = np.array(np.linspace(0, 10), dtype=np.float64).reshape(-1, 10)
    with pytest.raises(
        ValueError,
        match="Thetas should be a 1D array.",
    ):
        TwoPointReal(XY=real_two_point_xy, thetas=theta)


def test_two_point_from_metadata_cells(harmonic_two_point_xy, tp_factory):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    cells = TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy)
    two_point = TwoPoint.from_metadata([cells], tp_factory).pop()

    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    assert two_point.sacc_data_type == cells.get_sacc_name()

    assert isinstance(two_point.source0, SourceGalaxy)
    assert isinstance(two_point.source1, SourceGalaxy)

    assert_array_equal(two_point.source0.tracer_args.z, harmonic_two_point_xy.x.z)
    assert_array_equal(two_point.source1.tracer_args.z, harmonic_two_point_xy.y.z)

    assert_array_equal(two_point.source0.tracer_args.dndz, harmonic_two_point_xy.x.dndz)
    assert_array_equal(two_point.source1.tracer_args.dndz, harmonic_two_point_xy.y.dndz)


def test_two_point_from_metadata_cwindow(two_point_cwindow, tp_factory):
    two_point = TwoPoint.from_metadata([two_point_cwindow], tp_factory).pop()

    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    assert two_point.sacc_data_type == two_point_cwindow.get_sacc_name()

    assert isinstance(two_point.source0, SourceGalaxy)
    assert isinstance(two_point.source1, SourceGalaxy)

    assert_array_equal(two_point.source0.tracer_args.z, two_point_cwindow.XY.x.z)
    assert_array_equal(two_point.source1.tracer_args.z, two_point_cwindow.XY.y.z)

    assert_array_equal(two_point.source0.tracer_args.dndz, two_point_cwindow.XY.x.dndz)
    assert_array_equal(two_point.source1.tracer_args.dndz, two_point_cwindow.XY.y.dndz)


def test_two_point_from_metadata_xi_theta(real_two_point_xy, tp_factory):
    theta = np.array(np.linspace(0, 100, 100))
    xi_theta = TwoPointReal(XY=real_two_point_xy, thetas=theta)
    if xi_theta.get_sacc_name() == "galaxy_shear_xi_tt":
        return
    two_point = TwoPoint.from_metadata([xi_theta], tp_factory).pop()

    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    assert two_point.sacc_data_type == xi_theta.get_sacc_name()

    assert isinstance(two_point.source0, SourceGalaxy)
    assert isinstance(two_point.source1, SourceGalaxy)

    assert_array_equal(two_point.source0.tracer_args.z, real_two_point_xy.x.z)
    assert_array_equal(two_point.source1.tracer_args.z, real_two_point_xy.y.z)

    assert_array_equal(two_point.source0.tracer_args.dndz, real_two_point_xy.x.dndz)
    assert_array_equal(two_point.source1.tracer_args.dndz, real_two_point_xy.y.dndz)


def test_two_point_from_metadata_cells_unsupported_type(tp_factory):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={CMB.CONVERGENCE},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=CMB.CONVERGENCE, y_measurement=Galaxies.COUNTS
    )
    cells = TwoPointHarmonic(ells=ells, XY=xy)
    with pytest.raises(
        ValueError,
        match="No CMBConvergenceFactory found for type_source default.",
    ):
        TwoPoint.from_metadata([cells], tp_factory)


@pytest.fixture
def tp_factory_with_cmb():
    from firecrown.likelihood.two_point import TwoPointFactory, TwoPointCorrelationSpace
    from firecrown.likelihood.weak_lensing import WeakLensingFactory
    from firecrown.likelihood.number_counts import NumberCountsFactory

    return TwoPointFactory(
        correlation_space=TwoPointCorrelationSpace.HARMONIC,
        weak_lensing_factories=[WeakLensingFactory()],
        number_counts_factories=[NumberCountsFactory()],
        cmb_factories=[CMBConvergenceFactory()],
    )


def test_two_point_from_metadata_cmb_supported(tp_factory_with_cmb):
    """Test that CMB measurements work when CMB factory is provided."""
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={CMB.CONVERGENCE},
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=CMB.CONVERGENCE, y_measurement=Galaxies.COUNTS
    )
    cells = TwoPointHarmonic(ells=ells, XY=xy)

    # This should now work without raising an exception
    two_points = TwoPoint.from_metadata([cells], tp_factory_with_cmb)
    assert len(two_points) == 1
    two_point = two_points[0]
    assert two_point.sacc_data_type == cells.get_sacc_name()
