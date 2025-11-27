"""
Tests for the module firecrown.metadata_types and firecrown.metadata_functions.
"""

import pytest
import sacc
import numpy as np
from numpy.testing import assert_array_equal
from firecrown.metadata_types import (
    ALL_MEASUREMENTS,
    Clusters,
    CMB,
    Galaxies,
    InferredGalaxyZDist,
    TracerNames,
    TwoPointHarmonic,
    TwoPointXY,
    TwoPointReal,
)
from firecrown.metadata_functions import (
    make_two_point_xy,
    extract_all_real_metadata_indices,
    extract_all_harmonic_metadata_indices,
)
from firecrown.metadata_types._sacc_type_string import (
    _type_to_sacc_string_harmonic as harmonic,
    _type_to_sacc_string_real as real,
)
from firecrown.data_types import TwoPointMeasurement
from firecrown.likelihood._source import SourceGalaxy
from firecrown.likelihood._two_point import (
    TwoPoint,
    TwoPointFactory,
    TwoPointCorrelationSpace,
)
from firecrown.likelihood._cmb import CMBConvergenceFactory
from firecrown.likelihood._weak_lensing import WeakLensingFactory
from firecrown.likelihood.number_counts import NumberCountsFactory


def test_inferred_galaxy_z_dist():
    z_dist = InferredGalaxyZDist(
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    assert z_dist.bin_name == "b_name1"
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
            bin_name="b_name1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(101),
            measurements={Clusters.COUNTS},
        )


def test_inferred_galaxy_z_dist_bad_type():
    with pytest.raises(ValueError, match="The measurement should be a Measurement."):
        InferredGalaxyZDist(
            bin_name="b_name1",
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    with pytest.raises(
        ValueError,
        match="Measurement .* not in the measurements of b_name1.",
    ):
        TwoPointXY(
            x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
        )


def test_two_point_xy_gal_gal_invalid_y_measurement():
    x = InferredGalaxyZDist(
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    with pytest.raises(
        ValueError,
        match="Measurement .* not in the measurements of b_name2.",
    ):
        TwoPointXY(
            x=x, y=y, x_measurement=Galaxies.COUNTS, y_measurement=Galaxies.COUNTS
        )


def test_two_point_xy_cmb_gal():
    x = InferredGalaxyZDist(
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={CMB.CONVERGENCE},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=CMB.CONVERGENCE, y_measurement=Galaxies.COUNTS
    )
    assert xy.x == x
    assert xy.y == y
    assert xy.x_measurement == CMB.CONVERGENCE
    assert xy.y_measurement == Galaxies.COUNTS


def test_two_point_xy_invalid():
    x = InferredGalaxyZDist(
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
    )
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    with pytest.raises(
        ValueError,
        match="Measurements .* and .* must support harmonic-space calculations.",
    ):
        TwoPointHarmonic(XY=xy, ells=ells, window=weights, window_ells=window_ells)


def test_two_point_cwindow_invalid_window():
    x = InferredGalaxyZDist(
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_T, y_measurement=Galaxies.COUNTS
    )

    with pytest.raises(
        ValueError,
        match="window_ells must be None if window is None.",
    ):
        TwoPointHarmonic(XY=xy, ells=ells, window_ells=np.linspace(0, 1, 9))


def test_two_point_real():
    x = InferredGalaxyZDist(
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Galaxies.SHEAR_E, y_measurement=Galaxies.COUNTS
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
        harmonic(Galaxies.SHEAR_T, Galaxies.COUNTS)


def test_real_type_string_invalid():
    with pytest.raises(
        ValueError, match="Real-space correlation not supported for shear E."
    ):
        real(Galaxies.SHEAR_E, Galaxies.COUNTS)


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
    optimized_two_point_cwindow: TwoPointHarmonic,
):
    s = optimized_two_point_cwindow.to_yaml()
    recovered = TwoPointHarmonic.from_yaml(s)
    assert optimized_two_point_cwindow == recovered


def test_two_point_real_serialization(optimized_real_two_point_xy: TwoPointXY):
    theta = np.array(np.linspace(0, 10, 10))
    xi_theta = TwoPointReal(XY=optimized_real_two_point_xy, thetas=theta)
    s = xi_theta.to_yaml()
    recovered = TwoPointReal.from_yaml(s)
    assert xi_theta == recovered
    assert str(optimized_real_two_point_xy) == str(recovered.XY)
    assert str(xi_theta) == str(recovered)


def test_two_point_real_cmp_invalid(optimized_real_two_point_xy: TwoPointXY):
    theta = np.array(np.linspace(0, 10, 10))
    xi_theta = TwoPointReal(XY=optimized_real_two_point_xy, thetas=theta)
    with pytest.raises(
        ValueError,
        match="Can only compare TwoPointReal objects.",
    ):
        _ = xi_theta == "Im not a TwoPointReal"


def test_two_point_real_wrong_shape(optimized_real_two_point_xy: TwoPointXY):
    theta = np.array(np.linspace(0, 10), dtype=np.float64).reshape(-1, 10)
    with pytest.raises(
        ValueError,
        match="Thetas should be a 1D array.",
    ):
        TwoPointReal(XY=optimized_real_two_point_xy, thetas=theta)


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


def test_two_point_from_metadata_xi_theta(optimized_real_two_point_xy, tp_factory):
    theta = np.array(np.linspace(0, 100, 100))
    xi_theta = TwoPointReal(XY=optimized_real_two_point_xy, thetas=theta)
    if xi_theta.get_sacc_name() == "galaxy_shear_xi_tt":
        return
    two_point = TwoPoint.from_metadata([xi_theta], tp_factory).pop()

    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    assert two_point.sacc_data_type == xi_theta.get_sacc_name()

    assert isinstance(two_point.source0, SourceGalaxy)
    assert isinstance(two_point.source1, SourceGalaxy)

    assert_array_equal(two_point.source0.tracer_args.z, optimized_real_two_point_xy.x.z)
    assert_array_equal(two_point.source1.tracer_args.z, optimized_real_two_point_xy.y.z)

    assert_array_equal(
        two_point.source0.tracer_args.dndz, optimized_real_two_point_xy.x.dndz
    )
    assert_array_equal(
        two_point.source1.tracer_args.dndz, optimized_real_two_point_xy.y.dndz
    )


def test_two_point_from_metadata_cells_unsupported_type(tp_factory):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    x = InferredGalaxyZDist(
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Clusters.COUNTS},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    xy = TwoPointXY(
        x=x, y=y, x_measurement=Clusters.COUNTS, y_measurement=Galaxies.COUNTS
    )
    cells = TwoPointHarmonic(ells=ells, XY=xy)
    with pytest.raises(
        ValueError,
        match="Factory not found for measurement .*, it is not supported.",
    ):
        TwoPoint.from_metadata([cells], tp_factory)


@pytest.fixture(name="tp_factory_with_cmb")
def fixture_tp_factory_with_cmb():
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
        bin_name="b_name1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={CMB.CONVERGENCE},
    )
    y = InferredGalaxyZDist(
        bin_name="b_name2",
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
    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    # pylint: disable-next=no-member
    assert two_point.sacc_data_type == cells.get_sacc_name()


def test_make_two_point_xy_valid_galaxies():
    """Test make_two_point_xy with valid galaxy measurements."""
    x = InferredGalaxyZDist(
        bin_name="shear_bin_0",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    y = InferredGalaxyZDist(
        bin_name="shear_bin_1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    inferred_dict = {"shear_bin_0": x, "shear_bin_1": y}
    tracer_names = TracerNames("shear_bin_0", "shear_bin_1")
    data_type = "galaxy_shear_cl_ee"

    xy = make_two_point_xy(inferred_dict, tracer_names, data_type)

    assert xy.x == x
    assert xy.y == y
    assert xy.x_measurement == Galaxies.SHEAR_E
    assert xy.y_measurement == Galaxies.SHEAR_E


def test_make_two_point_xy_valid_cmb_galaxy():
    """Test make_two_point_xy with CMB-galaxy measurements."""
    cmb = InferredGalaxyZDist(
        bin_name="cmb_convergence",
        z=np.array([1100.0]),
        dndz=np.array([1.0]),
        measurements={CMB.CONVERGENCE},
    )
    galaxy = InferredGalaxyZDist(
        bin_name="galaxy_bin_0",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    inferred_dict = {"cmb_convergence": cmb, "galaxy_bin_0": galaxy}
    tracer_names = TracerNames("cmb_convergence", "galaxy_bin_0")
    data_type = harmonic(CMB.CONVERGENCE, Galaxies.COUNTS)

    xy = make_two_point_xy(inferred_dict, tracer_names, data_type)

    assert xy.x == cmb
    assert xy.y == galaxy
    assert xy.x_measurement == CMB.CONVERGENCE
    assert xy.y_measurement == Galaxies.COUNTS


def test_make_two_point_xy_valid_cmb_galaxy_needs_swap():
    """Test make_two_point_xy with CMB-galaxy measurements."""
    cmb = InferredGalaxyZDist(
        bin_name="cmb_convergence",
        z=np.array([1100.0]),
        dndz=np.array([1.0]),
        measurements={CMB.CONVERGENCE},
    )
    galaxy = InferredGalaxyZDist(
        bin_name="galaxy_bin_0",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},
    )
    inferred_dict = {"cmb_convergence": cmb, "galaxy_bin_0": galaxy}
    tracer_names = TracerNames("galaxy_bin_0", "cmb_convergence")
    data_type = harmonic(CMB.CONVERGENCE, Galaxies.COUNTS)
    # Even though the order is swapped, this should still work this behavior will be
    # removed in the future. It is kept for backwards compatibility and to avoid
    # breaking existing data files.
    xy = make_two_point_xy(inferred_dict, tracer_names, data_type)

    assert xy.x == cmb
    assert xy.y == galaxy
    assert xy.x_measurement == CMB.CONVERGENCE
    assert xy.y_measurement == Galaxies.COUNTS


def test_make_two_point_xy_missing_tracer_zdist():
    """Test make_two_point_xy raises exception when tracer zdist not found.

    Verifies that make_two_point_xy raises a ValueError with informative message
    when a requested tracer name is not in the inferred galaxy z distributions
    dictionary.
    """
    x = InferredGalaxyZDist(
        bin_name="shear_bin_0",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    # Only provide one tracer in the dictionary
    inferred_dict = {"shear_bin_0": x}
    # But request two tracers (second one doesn't exist)
    tracer_names = TracerNames("shear_bin_0", "shear_bin_1")
    data_type = harmonic(Galaxies.SHEAR_E, Galaxies.SHEAR_E)

    with pytest.raises(ValueError) as exc_info:
        make_two_point_xy(inferred_dict, tracer_names, data_type)

    error_msg = str(exc_info.value)
    assert "shear_bin_1" in error_msg
    assert "not found in inferred galaxy z distributions" in error_msg


def test_make_two_point_xy_missing_x_measurement():
    """Test make_two_point_xy when first tracer lacks required measurement."""
    x = InferredGalaxyZDist(
        bin_name="shear_bin_0",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},  # Has SHEAR_T but needs SHEAR_E
    )
    y = InferredGalaxyZDist(
        bin_name="shear_bin_1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    inferred_dict = {"shear_bin_0": x, "shear_bin_1": y}
    tracer_names = TracerNames("shear_bin_0", "shear_bin_1")
    data_type = harmonic(Galaxies.SHEAR_E, Galaxies.SHEAR_E)

    with pytest.raises(ValueError) as exc_info:
        make_two_point_xy(inferred_dict, tracer_names, data_type)

    error_msg = str(exc_info.value)
    assert "Tracer measurements do not match the SACC naming convention" in error_msg
    assert f"Data type: {data_type}" in error_msg
    assert (
        f"Expected measurements: ({Galaxies.SHEAR_E}, {Galaxies.SHEAR_E})" in error_msg
    )
    assert "shear_bin_0" in error_msg
    assert "shear_bin_1" in error_msg


def test_make_two_point_xy_missing_y_measurement():
    """Test make_two_point_xy when second tracer lacks required measurement."""
    x = InferredGalaxyZDist(
        bin_name="shear_bin_0",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},
    )
    y = InferredGalaxyZDist(
        bin_name="shear_bin_1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},  # Has SHEAR_T but needs SHEAR_E
    )
    inferred_dict = {"shear_bin_0": x, "shear_bin_1": y}
    tracer_names = TracerNames("shear_bin_0", "shear_bin_1")
    data_type = harmonic(Galaxies.SHEAR_E, Galaxies.SHEAR_E)

    with pytest.raises(ValueError) as exc_info:
        make_two_point_xy(inferred_dict, tracer_names, data_type)

    error_msg = str(exc_info.value)
    assert "Tracer measurements do not match the SACC naming convention" in error_msg
    assert f"Data type: {data_type}" in error_msg
    assert (
        f"Expected measurements: ({Galaxies.SHEAR_E}, {Galaxies.SHEAR_E})" in error_msg
    )
    assert "shear_bin_1" in error_msg


def test_make_two_point_xy_both_measurements_missing():
    """Test make_two_point_xy when both tracers lack required measurements."""
    x = InferredGalaxyZDist(
        bin_name="counts_bin_0",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.COUNTS},  # Has COUNTS, needs SHEAR_E
    )
    y = InferredGalaxyZDist(
        bin_name="shear_bin_1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_T},  # Has SHEAR_T, needs SHEAR_E
    )
    inferred_dict = {"counts_bin_0": x, "shear_bin_1": y}
    tracer_names = TracerNames("counts_bin_0", "shear_bin_1")
    data_type = harmonic(Galaxies.SHEAR_E, Galaxies.SHEAR_E)

    with pytest.raises(ValueError) as exc_info:
        make_two_point_xy(inferred_dict, tracer_names, data_type)

    error_msg = str(exc_info.value)
    assert "Tracer measurements do not match the SACC naming convention" in error_msg
    assert (
        f"Expected measurements: ({Galaxies.SHEAR_E}, {Galaxies.SHEAR_E})" in error_msg
    )


def test_make_two_point_xy_sacc_convention_explanation():
    """Test that error message includes SACC convention explanation."""
    x = InferredGalaxyZDist(
        bin_name="cmb_bin",
        z=np.array([1100.0]),
        dndz=np.array([1.0]),
        measurements={CMB.CONVERGENCE},
    )
    y = InferredGalaxyZDist(
        bin_name="galaxy_bin",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurements={Galaxies.SHEAR_E},  # Has SHEAR_E but needs COUNTS
    )
    inferred_dict = {"cmb_bin": x, "galaxy_bin": y}
    tracer_names = TracerNames("cmb_bin", "galaxy_bin")
    data_type = harmonic(CMB.CONVERGENCE, Galaxies.COUNTS)

    with pytest.raises(ValueError) as exc_info:
        make_two_point_xy(inferred_dict, tracer_names, data_type)

    error_msg = str(exc_info.value)
    # Check for convention explanation
    assert "According to the SACC convention" in error_msg
    assert "order of measurement types" in error_msg
    # Check for documentation link
    assert "sacc_usage.html" in error_msg


def test_extract_all_real_metadata_indices_no_swap(
    sacc_galaxy_xis_src0_lens0: tuple,
):
    """Test extract_all_real_metadata_indices with no tracer swap needed.

    This test verifies that when tracers are in the correct order according to
    the SACC convention (measurement type order matches tracer order), no swap
    is performed.
    """
    sacc_data, _, _, _ = sacc_galaxy_xis_src0_lens0

    indices = extract_all_real_metadata_indices(sacc_data)

    # Should have at least one real measurement (galaxy_shearDensity_xi_t)
    assert len(indices) > 0

    # Find the shear-density measurement
    shear_density_indices = [
        idx for idx in indices if "shearDensity" in idx["data_type"]
    ]
    assert len(shear_density_indices) > 0

    for idx in shear_density_indices:
        # The first tracer should be source (SHEAR_T) and second should be lens
        # (COUNTS)
        a, b = idx["tracer_types"]
        assert a == Galaxies.SHEAR_T
        assert b == Galaxies.COUNTS


def test_extract_all_real_metadata_indices_with_swap():
    """Test extract_all_real_metadata_indices when tracer swap is necessary.

    This test creates a SACC file where tracers are in reversed order compared
    to what the measurement type implies, triggering the swap logic.
    """
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 10)

    # Add tracers in "wrong" order for shear-density measurement
    # Normally: source (SHEAR_T) should come first, lens (COUNTS) second
    # But we add lens first, then source

    dndz_lens = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens_bin", z, dndz_lens)

    dndz_src = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "source_bin", z, dndz_src)

    # Add measurement: galaxy_shearDensity_xi_t expects (SHEAR_T, COUNTS)
    # But we provide it with (lens_bin, source_bin) which is (COUNTS, SHEAR_T)
    xis = np.random.normal(size=thetas.shape[0])
    # The only way to determine the incorrect order is to add autocorrelation
    sacc_data.add_theta_xi(
        "galaxy_shearDensity_xi_t", "lens_bin", "source_bin", thetas, xis
    )
    sacc_data.add_theta_xi("galaxy_density_xi", "lens_bin", "lens_bin", thetas, xis)
    sacc_data.add_theta_xi(
        "galaxy_shear_xi_plus", "source_bin", "source_bin", thetas, xis
    )

    cov = np.ones(30) * 0.01
    sacc_data.add_covariance(cov)

    with pytest.warns(
        DeprecationWarning,
        match="AUTO-CORRECTION PERFORMED",
    ):
        indices = extract_all_real_metadata_indices(sacc_data)

    # Should have the measurement
    assert len(indices) > 0

    # Find the shear-density measurement
    shear_density_indices = [
        idx for idx in indices if "shearDensity" in idx["data_type"]
    ]
    assert len(shear_density_indices) > 0

    for idx in shear_density_indices:
        # The swap should have corrected the order
        a, b = idx["tracer_types"]
        assert a == Galaxies.COUNTS
        assert b == Galaxies.SHEAR_T


def test_extract_all_real_metadata_indices_density_only():
    """Test extract_all_real_metadata_indices with density-only measurement."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 10)

    # Add lens tracers
    dndz0 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.3) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens1", z, dndz1)

    # Add density-density measurement
    xis = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi("galaxy_density_xi", "lens0", "lens1", thetas, xis)

    cov = np.diag(np.ones_like(xis) * 0.01)
    sacc_data.add_covariance(cov)

    indices = extract_all_real_metadata_indices(sacc_data)

    # Should have the measurement
    assert len(indices) > 0

    for idx in indices:
        if "density" in idx["data_type"]:
            a, b = idx["tracer_types"]
            assert a == Galaxies.COUNTS
            assert b == Galaxies.COUNTS


def test_extract_all_real_metadata_indices_shear_only():
    """Test extract_all_real_metadata_indices with shear-density measurement."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 10)

    # Add source and lens tracers
    dndz_src = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz_src)

    dndz_lens = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz_lens)

    # Add shear-density measurement (SHEAR_T, COUNTS)
    xis_t = np.random.normal(size=thetas.shape[0])
    sacc_data.add_theta_xi("galaxy_shearDensity_xi_t", "src0", "lens0", thetas, xis_t)

    cov = np.diag(np.ones_like(xis_t) * 0.01)
    sacc_data.add_covariance(cov)

    indices = extract_all_real_metadata_indices(sacc_data)

    # Should have the measurement
    assert len(indices) > 0

    for idx in indices:
        if "shearDensity" in idx["data_type"]:
            a, b = idx["tracer_types"]
            assert a == Galaxies.SHEAR_T
            assert b == Galaxies.COUNTS


def test_extract_all_real_metadata_indices_multiple_combinations():
    """Test extract_all_real_metadata_indices with multiple tracer combinations."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    thetas = np.linspace(0.0, 2.0 * np.pi, 10)

    # Add multiple source and lens tracers
    src_tracers = []
    for i in range(2):
        dndz = np.exp(-0.5 * (z - 0.5 + i * 0.1) ** 2 / 0.05 / 0.05)
        tracer_name = f"src{i}"
        sacc_data.add_tracer("NZ", tracer_name, z, dndz)
        src_tracers.append(tracer_name)

    lens_tracers = []
    for i in range(2):
        dndz = np.exp(-0.5 * (z - 0.1 + i * 0.1) ** 2 / 0.05 / 0.05)
        tracer_name = f"lens{i}"
        sacc_data.add_tracer("NZ", tracer_name, z, dndz)
        lens_tracers.append(tracer_name)

    # Add cross-correlations: shear-density measurements
    data_list = []
    for src in src_tracers:
        for lens in lens_tracers:
            xis = np.random.normal(size=thetas.shape[0])
            sacc_data.add_theta_xi("galaxy_shearDensity_xi_t", src, lens, thetas, xis)
            data_list.append(xis)

    cov_data = np.concatenate(data_list)
    cov = np.diag(np.ones_like(cov_data) * 0.01)
    sacc_data.add_covariance(cov)

    indices = extract_all_real_metadata_indices(sacc_data)

    # Should have multiple measurements
    assert len(indices) == len(src_tracers) * len(lens_tracers)

    # All should be shear-density measurements with correct type order
    for idx in indices:
        a, b = idx["tracer_types"]
        assert a == Galaxies.SHEAR_T
        assert b == Galaxies.COUNTS


def test_extract_all_harmonic_metadata_indices_no_swap(
    sacc_galaxy_cells_src0_lens0: tuple,
):
    """Test extract_all_harmonic_metadata_indices with no tracer swap needed.

    This test verifies that when tracers are in the correct order according to
    the SACC convention (measurement type order matches tracer order), no swap
    is performed.
    """
    sacc_data, _, _, _ = sacc_galaxy_cells_src0_lens0

    indices = extract_all_harmonic_metadata_indices(sacc_data)

    # Should have at least one harmonic measurement (galaxy_shearDensity_cl_e)
    assert len(indices) > 0

    # Find the shear-density measurement
    shear_density_indices = [
        idx for idx in indices if "shearDensity" in idx["data_type"]
    ]
    assert len(shear_density_indices) > 0

    for idx in shear_density_indices:
        # The first tracer should be source (SHEAR_E) and second should be lens (COUNTS)
        a, b = idx["tracer_types"]
        assert a == Galaxies.SHEAR_E
        assert b == Galaxies.COUNTS


def test_extract_all_harmonic_metadata_indices_with_swap():
    """Test extract_all_harmonic_metadata_indices when tracer swap is necessary.

    This test creates a SACC file where tracers are in reversed order compared
    to what the measurement type implies, triggering the swap logic.
    """
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    # Add tracers in "wrong" order for shear-density measurement
    # Normally: source (SHEAR_E) should come first, lens (COUNTS) second
    # But we add lens first, then source

    dndz_lens = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens_bin", z, dndz_lens)

    dndz_src = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "source_bin", z, dndz_src)

    # Add measurement: galaxy_shearDensity_cl_e expects (SHEAR_E, COUNTS)
    # But we provide it with (lens_bin, source_bin) which is (COUNTS, SHEAR_E)
    Cells = np.random.normal(size=ells.shape[0])
    # The only way to determine the incorrect order is to add autocorrelation
    sacc_data.add_ell_cl(
        "galaxy_shearDensity_cl_e", "lens_bin", "source_bin", ells, Cells
    )
    sacc_data.add_ell_cl("galaxy_density_cl", "lens_bin", "lens_bin", ells, Cells)
    sacc_data.add_ell_cl("galaxy_shear_cl_ee", "source_bin", "source_bin", ells, Cells)

    cov = np.ones(30) * 0.01
    sacc_data.add_covariance(cov)

    with pytest.warns(
        DeprecationWarning,
        match="AUTO-CORRECTION PERFORMED",
    ):
        indices = extract_all_harmonic_metadata_indices(sacc_data)

    # Should have the measurement
    assert len(indices) > 0

    # Find the shear-density measurement
    shear_density_indices = [
        idx for idx in indices if "shearDensity" in idx["data_type"]
    ]
    assert len(shear_density_indices) > 0

    for idx in shear_density_indices:
        # The swap should have corrected the order
        a, b = idx["tracer_types"]
        assert a == Galaxies.COUNTS
        assert b == Galaxies.SHEAR_E


def test_extract_all_harmonic_metadata_indices_density_only():
    """Test extract_all_harmonic_metadata_indices with density-only measurement."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    # Add lens tracers
    dndz0 = np.exp(-0.5 * (z - 0.1) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.3) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "lens1", z, dndz1)

    # Add density-density measurement
    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_density_cl", "lens0", "lens1", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    indices = extract_all_harmonic_metadata_indices(sacc_data)

    # Should have the measurement
    assert len(indices) > 0

    for idx in indices:
        if "density" in idx["data_type"]:
            a, b = idx["tracer_types"]
            assert a == Galaxies.COUNTS
            assert b == Galaxies.COUNTS


def test_extract_all_harmonic_metadata_indices_shear_only():
    """Test extract_all_harmonic_metadata_indices with shear-only measurement."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    # Add source tracers
    dndz0 = np.exp(-0.5 * (z - 0.5) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src0", z, dndz0)

    dndz1 = np.exp(-0.5 * (z - 0.6) ** 2 / 0.05 / 0.05)
    sacc_data.add_tracer("NZ", "src1", z, dndz1)

    # Add shear-shear measurement (both tracers have SHEAR_E)
    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_shear_cl_ee", "src0", "src1", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    indices = extract_all_harmonic_metadata_indices(sacc_data)

    # Should have the measurement
    assert len(indices) > 0

    for idx in indices:
        if "shear" in idx["data_type"]:
            a, b = idx["tracer_types"]
            assert a == Galaxies.SHEAR_E
            assert b == Galaxies.SHEAR_E


def test_extract_all_harmonic_metadata_indices_multiple_combinations():
    """Test extract_all_harmonic_metadata_indices with multiple tracer combinations."""
    sacc_data = sacc.Sacc()

    z = np.linspace(0, 1.0, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    # Add multiple source and lens tracers
    src_tracers = []
    for i in range(2):
        dndz = np.exp(-0.5 * (z - 0.5 + i * 0.1) ** 2 / 0.05 / 0.05)
        tracer_name = f"src{i}"
        sacc_data.add_tracer("NZ", tracer_name, z, dndz)
        src_tracers.append(tracer_name)

    lens_tracers = []
    for i in range(2):
        dndz = np.exp(-0.5 * (z - 0.1 + i * 0.1) ** 2 / 0.05 / 0.05)
        tracer_name = f"lens{i}"
        sacc_data.add_tracer("NZ", tracer_name, z, dndz)
        lens_tracers.append(tracer_name)

    # Add cross-correlations: shear-density measurements
    data_list = []
    for src in src_tracers:
        for lens in lens_tracers:
            Cells = np.random.normal(size=ells.shape[0])
            sacc_data.add_ell_cl("galaxy_shearDensity_cl_e", src, lens, ells, Cells)
            data_list.append(Cells)

    cov_data = np.concatenate(data_list)
    cov = np.diag(np.ones_like(cov_data) * 0.01)
    sacc_data.add_covariance(cov)

    indices = extract_all_harmonic_metadata_indices(sacc_data)

    # Should have multiple measurements
    assert len(indices) == len(src_tracers) * len(lens_tracers)

    # All should be shear-density measurements with correct type order
    for idx in indices:
        a, b = idx["tracer_types"]
        assert a == Galaxies.SHEAR_E
        assert b == Galaxies.COUNTS
