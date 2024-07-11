"""
Tests for the module firecrown.metadata.two_point
"""

from itertools import product, chain
from unittest.mock import MagicMock
import pytest
import numpy as np
from numpy.testing import assert_array_equal

import sacc_name_mapping as snm
from firecrown.metadata.two_point_types import compare_enums
from firecrown.metadata.two_point import (
    ALL_MEASUREMENTS,
    Clusters,
    CMB,
    Galaxies,
    InferredGalaxyZDist,
    measurement_is_compatible as is_compatible,
    measurement_is_compatible_real as is_compatible_real,
    measurement_is_compatible_harmonic as is_compatible_harmonic,
    measurement_supports_harmonic as supports_harmonic,
    measurement_supports_real as supports_real,
    TracerNames,
    TwoPointCells,
    TwoPointCWindow,
    TwoPointXY,
    TwoPointXiTheta,
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
    Window,
)
from firecrown.likelihood.source import SourceGalaxy
from firecrown.likelihood.two_point import TwoPoint


def test_order_enums():
    assert compare_enums(CMB.CONVERGENCE, Clusters.COUNTS) < 0
    assert compare_enums(Clusters.COUNTS, CMB.CONVERGENCE) > 0

    assert compare_enums(CMB.CONVERGENCE, Galaxies.COUNTS) < 0
    assert compare_enums(Galaxies.COUNTS, CMB.CONVERGENCE) > 0

    assert compare_enums(Galaxies.SHEAR_E, Galaxies.SHEAR_T) < 0
    assert compare_enums(Galaxies.SHEAR_E, Galaxies.COUNTS) < 0
    assert compare_enums(Galaxies.SHEAR_T, Galaxies.COUNTS) < 0

    assert compare_enums(Galaxies.COUNTS, Galaxies.SHEAR_E) > 0

    for enumerand in ALL_MEASUREMENTS:
        assert compare_enums(enumerand, enumerand) == 0


def test_enumeration_equality_galaxy():
    for e1, e2 in product(Galaxies, chain(CMB, Clusters)):
        assert e1 != e2


def test_enumeration_equality_cmb():
    for e1, e2 in product(CMB, chain(Galaxies, Clusters)):
        assert e1 != e2


def test_enumeration_equality_cluster():
    for e1, e2 in product(Clusters, chain(CMB, Galaxies)):
        assert e1 != e2


def test_exact_matches():
    for sacc_name, space, (enum_1, enum_2) in snm.mappings:
        if space == "ell":
            assert harmonic(enum_1, enum_2) == sacc_name
        elif space == "theta":
            assert real(enum_1, enum_2) == sacc_name
        else:
            raise ValueError(f"Illegal 'space' value {space} in testing data")


def test_translation_invariants():
    for a, b in product(ALL_MEASUREMENTS, ALL_MEASUREMENTS):
        assert isinstance(a, (Galaxies, CMB, Clusters))
        assert isinstance(b, (Galaxies, CMB, Clusters))
        if is_compatible_real(a, b):
            assert real(a, b) == real(b, a)
        if is_compatible_harmonic(a, b):
            assert harmonic(a, b) == harmonic(b, a)
        if (
            supports_harmonic(a)
            and supports_harmonic(b)
            and supports_real(a)
            and supports_real(b)
        ):
            assert harmonic(a, b) != real(a, b)


def test_unsupported_type_galaxy():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Untranslated Galaxy Measurement encountered"):
        Galaxies.sacc_measurement_name(unknown_type)

    with pytest.raises(ValueError, match="Untranslated Galaxy Measurement encountered"):
        Galaxies.polarization(unknown_type)


def test_unsupported_type_cmb():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Untranslated CMBMeasurement encountered"):
        CMB.sacc_measurement_name(unknown_type)

    with pytest.raises(ValueError, match="Untranslated CMBMeasurement encountered"):
        CMB.polarization(unknown_type)


def test_unsupported_type_cluster():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Untranslated ClusterMeasurement encountered"):
        Clusters.sacc_measurement_name(unknown_type)

    with pytest.raises(ValueError, match="Untranslated ClusterMeasurement encountered"):
        Clusters.polarization(unknown_type)


def test_type_hashs():
    for e1, e2 in product(ALL_MEASUREMENTS, ALL_MEASUREMENTS):
        if e1 == e2:
            assert hash(e1) == hash(e2)
        else:
            assert hash(e1) != hash(e2)


def test_measurement_is_compatible():
    for a, b in product(ALL_MEASUREMENTS, ALL_MEASUREMENTS):
        assert isinstance(a, (Galaxies, CMB, Clusters))
        assert isinstance(b, (Galaxies, CMB, Clusters))
        if is_compatible_real(a, b) or is_compatible_harmonic(a, b):
            assert is_compatible(a, b)
        else:
            assert not is_compatible(a, b)


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
        match="Measurement Galaxies.COUNTS not in the measurements of bname1.",
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
        match="Measurement Galaxies.COUNTS not in the measurements of bname2.",
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


def test_two_point_cells():
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
    cells = TwoPointCells(ells=ells, XY=xy)

    assert_array_equal(cells.ells, ells)
    assert cells.XY == xy
    assert cells.get_sacc_name() == harmonic(xy.x_measurement, xy.y_measurement)


def test_two_point_cells_invalid_ells():
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
        TwoPointCells(ells=ells, XY=xy)


def test_two_point_cells_invalid_type():
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
        TwoPointCells(ells=ells, XY=xy)


def test_window_equality():
    # Construct two windows with the same parameters, but different objects.
    window_1 = Window(
        ells=np.array(np.linspace(0, 100, 100), dtype=np.int64),
        weights=np.ones(400).reshape(-1, 4),
        ells_for_interpolation=np.array(np.linspace(0, 100, 100), dtype=np.int64),
    )
    window_2 = Window(
        ells=np.array(np.linspace(0, 100, 100), dtype=np.int64),
        weights=np.ones(400).reshape(-1, 4),
        ells_for_interpolation=np.array(np.linspace(0, 100, 100), dtype=np.int64),
    )
    # Two windows constructed from the same parameters should be equal.
    assert window_1 == window_2

    # If we get rid of the ells_for_interpolation from one, they should no
    # longer be equal.
    window_2.ells_for_interpolation = None
    assert window_1 != window_2
    assert window_2 != window_1

    # If we have nulled out both ells_for_interpolation, they should be equal
    # again.
    window_1.ells_for_interpolation = None
    assert window_1 == window_2

    # And if we change the ells, they should no longer be equal.
    window_2.ells[0] = window_2.ells[0] + 1
    assert window_1 != window_2


def test_two_point_window():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    ells_for_interpolation = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

    window = Window(
        ells=ells,
        weights=weights,
        ells_for_interpolation=ells_for_interpolation,
    )

    assert_array_equal(window.ells, ells)
    assert_array_equal(window.weights, weights)
    assert window.ells_for_interpolation is not None
    assert_array_equal(window.ells_for_interpolation, ells_for_interpolation)
    assert window.n_observations() == 4


def test_two_point_window_invalid_ells():
    ells = np.array(np.linspace(0, 100), dtype=np.int64).reshape(-1, 10)
    weights = np.ones(400).reshape(-1, 4)
    ells_for_interpolation = np.array(np.linspace(0, 100, 100), dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="Ells should be a 1D array.",
    ):
        Window(
            ells=ells,
            weights=weights,
            ells_for_interpolation=ells_for_interpolation,
        )


def test_two_point_window_invalid_weights():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 5)
    ells_for_interpolation = np.array(np.linspace(0, 100, 100), dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="Weights should have the same number of rows as ells.",
    ):
        Window(
            ells=ells,
            weights=weights,
            ells_for_interpolation=ells_for_interpolation,
        )


def test_two_point_window_invalid_ells_for_interpolation():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)
    ells_for_interpolation = np.array(np.linspace(0, 100), dtype=np.int64).reshape(
        -1, 10
    )

    with pytest.raises(
        ValueError,
        match="Ells for interpolation should be a 1D array.",
    ):
        Window(
            ells=ells,
            weights=weights,
            ells_for_interpolation=ells_for_interpolation,
        )


def test_two_point_window_invalid_weights_shape():
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400)
    ells_for_interpolation = np.array(np.linspace(0, 100), dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="Weights should be a 2D array.",
    ):
        Window(
            ells=ells,
            weights=weights,
            ells_for_interpolation=ells_for_interpolation,
        )


def test_two_point_two_point_cwindow():
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
    two_point = TwoPointCWindow(XY=xy, window=window)

    assert two_point.window == window
    assert two_point.XY == xy
    assert two_point.get_sacc_name() == harmonic(xy.x_measurement, xy.y_measurement)


def test_two_point_two_point_cwindow_invalid():
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
        TwoPointCWindow(XY=xy, window=window)


def test_two_point_two_point_cwindow_invalid_window():
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
        match="Window should be a Window object.",
    ):
        TwoPointCWindow(XY=xy, window="Im not a window")  # type: ignore


def test_two_point_xi_theta():

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
    two_point = TwoPointXiTheta(XY=xy, thetas=theta)

    assert_array_equal(two_point.thetas, theta)
    assert two_point.XY == xy
    assert two_point.get_sacc_name() == real(xy.x_measurement, xy.y_measurement)


def test_two_point_xi_theta_invalid():
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
        TwoPointXiTheta(XY=xy, thetas=theta)


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


def test_two_point_xy_str(
    harmonic_bin_1: InferredGalaxyZDist, harmonic_bin_2: InferredGalaxyZDist
):
    xy = TwoPointXY(
        x=harmonic_bin_1,
        y=harmonic_bin_2,
        x_measurement=list(harmonic_bin_1.measurements)[0],
        y_measurement=list(harmonic_bin_2.measurements)[0],
    )
    assert str(xy) == f"({harmonic_bin_1.bin_name}, {harmonic_bin_2.bin_name})"


def test_two_point_xy_serialization(
    harmonic_bin_1: InferredGalaxyZDist, harmonic_bin_2: InferredGalaxyZDist
):
    xy = TwoPointXY(
        x=harmonic_bin_1,
        y=harmonic_bin_2,
        x_measurement=list(harmonic_bin_1.measurements)[0],
        y_measurement=list(harmonic_bin_2.measurements)[0],
    )
    s = xy.to_yaml()
    # Take a look at how hideous the generated string
    # is.
    recovered = TwoPointXY.from_yaml(s)
    assert xy == recovered
    assert str(xy) == str(recovered)


def test_two_point_cells_str(
    harmonic_bin_1: InferredGalaxyZDist, harmonic_bin_2: InferredGalaxyZDist
):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    xy = TwoPointXY(
        x=harmonic_bin_1,
        y=harmonic_bin_2,
        x_measurement=list(harmonic_bin_1.measurements)[0],
        y_measurement=list(harmonic_bin_2.measurements)[0],
    )
    cells = TwoPointCells(ells=ells, XY=xy)
    assert str(cells) == f"{str(xy)}[{cells.get_sacc_name()}]"


def test_two_point_cells_serialization(
    harmonic_bin_1: InferredGalaxyZDist, harmonic_bin_2: InferredGalaxyZDist
):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    xy = TwoPointXY(
        x=harmonic_bin_1,
        y=harmonic_bin_2,
        x_measurement=list(harmonic_bin_1.measurements)[0],
        y_measurement=list(harmonic_bin_2.measurements)[0],
    )
    cells = TwoPointCells(ells=ells, XY=xy)
    s = cells.to_yaml()
    recovered = TwoPointCells.from_yaml(s)
    assert cells == recovered
    assert str(xy) == str(recovered.XY)
    assert str(cells) == str(recovered)


def test_window_serialization(window_1: Window):
    s = window_1.to_yaml()
    recovered = Window.from_yaml(s)
    assert window_1 == recovered


def test_two_point_cwindow_serialization(two_point_cwindow_1: TwoPointCWindow):
    s = two_point_cwindow_1.to_yaml()
    recovered = TwoPointCWindow.from_yaml(s)
    assert two_point_cwindow_1 == recovered


def test_two_point_xi_theta_serialization(
    real_bin_1: InferredGalaxyZDist, real_bin_2: InferredGalaxyZDist
):
    xy = TwoPointXY(
        x=real_bin_1,
        y=real_bin_2,
        x_measurement=list(real_bin_1.measurements)[0],
        y_measurement=list(real_bin_2.measurements)[0],
    )

    theta = np.array(np.linspace(0, 10, 10))
    xi_theta = TwoPointXiTheta(XY=xy, thetas=theta)
    s = xi_theta.to_yaml()
    recovered = TwoPointXiTheta.from_yaml(s)
    assert xi_theta == recovered
    assert str(xy) == str(recovered.XY)
    assert str(xi_theta) == str(recovered)


def test_two_point_from_metadata_cells(
    harmonic_bin_1, harmonic_bin_2, wl_factory, nc_factory
):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    xy = TwoPointXY(
        x=harmonic_bin_1,
        y=harmonic_bin_2,
        x_measurement=list(harmonic_bin_1.measurements)[0],
        y_measurement=list(harmonic_bin_2.measurements)[0],
    )
    cells = TwoPointCells(ells=ells, XY=xy)
    two_point = TwoPoint.from_metadata_harmonic([cells], wl_factory, nc_factory).pop()

    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    assert two_point.sacc_data_type == cells.get_sacc_name()

    assert isinstance(two_point.source0, SourceGalaxy)
    assert isinstance(two_point.source1, SourceGalaxy)

    assert_array_equal(two_point.source0.tracer_args.z, harmonic_bin_1.z)
    assert_array_equal(two_point.source0.tracer_args.z, harmonic_bin_1.z)

    assert_array_equal(two_point.source0.tracer_args.dndz, harmonic_bin_1.dndz)
    assert_array_equal(two_point.source1.tracer_args.dndz, harmonic_bin_2.dndz)


def test_two_point_from_metadata_cwindow(two_point_cwindow_1, wl_factory, nc_factory):
    two_point = TwoPoint.from_metadata_harmonic(
        [two_point_cwindow_1], wl_factory, nc_factory
    ).pop()

    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    assert two_point.sacc_data_type == two_point_cwindow_1.get_sacc_name()

    assert isinstance(two_point.source0, SourceGalaxy)
    assert isinstance(two_point.source1, SourceGalaxy)

    assert_array_equal(two_point.source0.tracer_args.z, two_point_cwindow_1.XY.x.z)
    assert_array_equal(two_point.source1.tracer_args.z, two_point_cwindow_1.XY.y.z)

    assert_array_equal(
        two_point.source0.tracer_args.dndz, two_point_cwindow_1.XY.x.dndz
    )
    assert_array_equal(
        two_point.source1.tracer_args.dndz, two_point_cwindow_1.XY.y.dndz
    )


def test_two_point_from_metadata_xi_theta(
    real_bin_1, real_bin_2, wl_factory, nc_factory
):
    theta = np.array(np.linspace(0, 100, 100))
    xy = TwoPointXY(
        x=real_bin_1,
        y=real_bin_2,
        x_measurement=list(real_bin_1.measurements)[0],
        y_measurement=list(real_bin_2.measurements)[0],
    )
    xi_theta = TwoPointXiTheta(XY=xy, thetas=theta)
    if xi_theta.get_sacc_name() == "galaxy_shear_xi_tt":
        return
    two_point = TwoPoint.from_metadata_real([xi_theta], wl_factory, nc_factory).pop()

    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    assert two_point.sacc_data_type == xi_theta.get_sacc_name()

    assert isinstance(two_point.source0, SourceGalaxy)
    assert isinstance(two_point.source1, SourceGalaxy)

    assert_array_equal(two_point.source0.tracer_args.z, real_bin_1.z)
    assert_array_equal(two_point.source1.tracer_args.z, real_bin_2.z)

    assert_array_equal(two_point.source0.tracer_args.dndz, real_bin_1.dndz)
    assert_array_equal(two_point.source1.tracer_args.dndz, real_bin_2.dndz)


def test_two_point_from_metadata_cells_unsupported_type(wl_factory, nc_factory):
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
    cells = TwoPointCells(ells=ells, XY=xy)
    with pytest.raises(
        ValueError,
        match="Measurement .* not supported!",
    ):
        TwoPoint.from_metadata_harmonic([cells], wl_factory, nc_factory)
