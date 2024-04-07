"""
Tests for the module firecrown.metadata.two_point
"""

from itertools import product, chain
from unittest.mock import MagicMock
import pytest
import numpy as np
from numpy.testing import assert_array_equal

import sacc_name_mapping as snm
from firecrown.metadata.two_point import (
    ALL_MEASURED_TYPES,
    ClusterMeasuredType,
    CMBMeasuredType,
    compare_enums,
    GalaxyMeasuredType,
    InferredGalaxyZDist,
    measured_type_is_compatible as is_compatible,
    measured_type_supports_harmonic as supports_harmonic,
    measured_type_supports_real as supports_real,
    TwoPointCells,
    TwoPointCWindow,
    TwoPointXY,
    TwoPointXiTheta,
    type_to_sacc_string_harmonic as harmonic,
    type_to_sacc_string_real as real,
    Window,
)


def test_order_enums():
    assert compare_enums(CMBMeasuredType.CONVERGENCE, ClusterMeasuredType.COUNTS) < 0
    assert compare_enums(ClusterMeasuredType.COUNTS, CMBMeasuredType.CONVERGENCE) > 0

    assert compare_enums(CMBMeasuredType.CONVERGENCE, GalaxyMeasuredType.COUNTS) < 0
    assert compare_enums(GalaxyMeasuredType.COUNTS, CMBMeasuredType.CONVERGENCE) > 0

    assert compare_enums(GalaxyMeasuredType.SHEAR_E, GalaxyMeasuredType.SHEAR_T) < 0
    assert compare_enums(GalaxyMeasuredType.SHEAR_E, GalaxyMeasuredType.COUNTS) < 0
    assert compare_enums(GalaxyMeasuredType.SHEAR_T, GalaxyMeasuredType.COUNTS) < 0

    assert compare_enums(GalaxyMeasuredType.COUNTS, GalaxyMeasuredType.SHEAR_E) > 0

    for enumerand in ALL_MEASURED_TYPES:
        assert compare_enums(enumerand, enumerand) == 0


def test_enumeration_equality_galaxy():
    for e1, e2 in product(
        GalaxyMeasuredType, chain(CMBMeasuredType, ClusterMeasuredType)
    ):
        assert e1 != e2


def test_enumeration_equality_cmb():
    for e1, e2 in product(
        CMBMeasuredType, chain(GalaxyMeasuredType, ClusterMeasuredType)
    ):
        assert e1 != e2


def test_enumeration_equality_cluster():
    for e1, e2 in product(
        ClusterMeasuredType, chain(CMBMeasuredType, GalaxyMeasuredType)
    ):
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
    for a, b in product(ALL_MEASURED_TYPES, ALL_MEASURED_TYPES):
        assert isinstance(a, (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType))
        assert isinstance(b, (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType))
        if supports_real(a) and supports_real(b):
            assert real(a, b) == real(b, a)
        if supports_harmonic(a) and supports_harmonic(b):
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

    with pytest.raises(ValueError, match="Untranslated GalaxyMeasuredType encountered"):
        GalaxyMeasuredType.sacc_measurement_name(unknown_type)

    with pytest.raises(ValueError, match="Untranslated GalaxyMeasuredType encountered"):
        GalaxyMeasuredType.polarization(unknown_type)


def test_unsupported_type_cmb():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Untranslated CMBMeasuredType encountered"):
        CMBMeasuredType.sacc_measurement_name(unknown_type)

    with pytest.raises(ValueError, match="Untranslated CMBMeasuredType encountered"):
        CMBMeasuredType.polarization(unknown_type)


def test_unsupported_type_cluster():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(
        ValueError, match="Untranslated ClusterMeasuredType encountered"
    ):
        ClusterMeasuredType.sacc_measurement_name(unknown_type)

    with pytest.raises(
        ValueError, match="Untranslated ClusterMeasuredType encountered"
    ):
        ClusterMeasuredType.polarization(unknown_type)


def test_type_hashs():
    for e1, e2 in product(ALL_MEASURED_TYPES, ALL_MEASURED_TYPES):
        if e1 == e2:
            assert hash(e1) == hash(e2)
        else:
            assert hash(e1) != hash(e2)


def test_measured_type_is_compatible():
    for a, b in product(ALL_MEASURED_TYPES, ALL_MEASURED_TYPES):
        assert isinstance(a, (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType))
        assert isinstance(b, (GalaxyMeasuredType, CMBMeasuredType, ClusterMeasuredType))
        if (supports_real(a) and supports_real(b)) or (
            supports_harmonic(a) and supports_harmonic(b)
        ):
            assert is_compatible(a, b)
        else:
            assert not is_compatible(a, b)


def test_inferred_galaxy_z_dist():
    z_dist = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    assert z_dist.bin_name == "bname1"
    assert z_dist.z[0] == 0
    assert z_dist.z[-1] == 1
    assert z_dist.dndz[0] == 1
    assert z_dist.dndz[-1] == 1
    assert z_dist.measured_type == GalaxyMeasuredType.COUNTS


def test_inferred_galaxy_z_dist_bad_shape():
    with pytest.raises(
        ValueError, match="The z and dndz arrays should have the same shape."
    ):
        InferredGalaxyZDist(
            bin_name="bname1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(101),
            measured_type=ClusterMeasuredType.COUNTS,
        )


def test_inferred_galaxy_z_dist_bad_type():
    with pytest.raises(ValueError, match="The measured_type should be a MeasuredType."):
        InferredGalaxyZDist(
            bin_name="bname1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measured_type=0,  # type: ignore
        )


def test_inferred_galaxy_z_dist_bad_name():
    with pytest.raises(ValueError, match="The bin_name should not be empty."):
        InferredGalaxyZDist(
            bin_name="",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measured_type=GalaxyMeasuredType.COUNTS,
        )


def test_two_point_xy_gal_gal():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    assert xy.x == x
    assert xy.y == y


def test_two_point_xy_cmb_gal():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=CMBMeasuredType.CONVERGENCE,
    )
    xy = TwoPointXY(x=x, y=y)
    assert xy.x == x
    assert xy.y == y


def test_two_point_xy_invalid():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.SHEAR_E,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.SHEAR_T,
    )
    with pytest.raises(
        ValueError,
        match=("Measured types .* and .* are not compatible."),
    ):
        TwoPointXY(x=x, y=y)


def test_two_point_cells():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    cells = TwoPointCells(ells=ells, XY=xy)

    assert_array_equal(cells.ells, ells)
    assert cells.XY == xy
    assert cells.get_sacc_name() == harmonic(x.measured_type, y.measured_type)


def test_two_point_cells_invalid_ells():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
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
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.SHEAR_T,
    )
    xy = TwoPointXY(x=x, y=y)
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    with pytest.raises(
        ValueError,
        match="Measured types .* and .* must support harmonic-space calculations.",
    ):
        TwoPointCells(ells=ells, XY=xy)


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
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    two_point = TwoPointCWindow(XY=xy, window=window)

    assert two_point.window == window
    assert two_point.XY == xy
    assert two_point.get_sacc_name() == harmonic(x.measured_type, y.measured_type)


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
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.SHEAR_T,
    )
    xy = TwoPointXY(x=x, y=y)
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    with pytest.raises(
        ValueError,
        match="Measured types .* and .* must support harmonic-space calculations.",
    ):
        TwoPointCWindow(XY=xy, window=window)


def test_two_point_two_point_cwindow_invalid_window():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.SHEAR_T,
    )
    xy = TwoPointXY(x=x, y=y)
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
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    theta = np.array(np.linspace(0, 100, 100))
    two_point = TwoPointXiTheta(XY=xy, thetas=theta)

    assert_array_equal(two_point.thetas, theta)
    assert two_point.XY == xy
    assert two_point.get_sacc_name() == real(x.measured_type, y.measured_type)


def test_two_point_xi_theta_invalid():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measured_type=GalaxyMeasuredType.SHEAR_E,
    )
    xy = TwoPointXY(x=x, y=y)
    theta = np.array(np.linspace(0, 100, 100))
    with pytest.raises(
        ValueError,
        match="Measured types .* and .* must support real-space calculations.",
    ):
        TwoPointXiTheta(XY=xy, thetas=theta)
