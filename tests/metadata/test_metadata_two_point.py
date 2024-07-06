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
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood.two_point import TwoPoint


@pytest.fixture(
    name="harmonic_bin_1",
    params=[Galaxies.COUNTS, Galaxies.SHEAR_E],
)
def make_harmonic_bin_1(request) -> InferredGalaxyZDist:
    """Generate an InferredGalaxyZDist object with 5 bins."""
    x = InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.linspace(0, 1, 5),
        dndz=np.array([0.1, 0.5, 0.2, 0.3, 0.4]),
        measurement=request.param,
    )
    return x


@pytest.fixture(
    name="harmonic_bin_2",
    params=[Galaxies.COUNTS, Galaxies.SHEAR_E],
)
def make_harmonic_bin_2(request) -> InferredGalaxyZDist:
    """Generate an InferredGalaxyZDist object with 3 bins."""
    x = InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.linspace(0, 1, 3),
        dndz=np.array([0.1, 0.5, 0.4]),
        measurement=request.param,
    )
    return x


@pytest.fixture(
    name="real_bin_1",
    params=[Galaxies.COUNTS, Galaxies.SHEAR_T],
)
def make_real_bin_1(request) -> InferredGalaxyZDist:
    """Generate an InferredGalaxyZDist object with 5 bins."""
    x = InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.linspace(0, 1, 5),
        dndz=np.array([0.1, 0.5, 0.2, 0.3, 0.4]),
        measurement=request.param,
    )
    return x


@pytest.fixture(
    name="real_bin_2",
    params=[Galaxies.COUNTS, Galaxies.SHEAR_T],
)
def make_real_bin_2(request) -> InferredGalaxyZDist:
    """Generate an InferredGalaxyZDist object with 3 bins."""
    x = InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.linspace(0, 1, 3),
        dndz=np.array([0.1, 0.5, 0.4]),
        measurement=request.param,
    )
    return x


@pytest.fixture(name="window_1")
def make_window_1() -> Window:
    """Generate a Window object with 100 ells."""
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    ells_for_interpolation = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    weights = np.ones(400).reshape(-1, 4)

    window = Window(
        ells=ells,
        weights=weights,
        ells_for_interpolation=ells_for_interpolation,
    )
    return window


@pytest.fixture(name="two_point_cwindow_1")
def make_two_point_cwindow_1(
    window_1: Window,
    harmonic_bin_1: InferredGalaxyZDist,
    harmonic_bin_2: InferredGalaxyZDist,
) -> TwoPointCWindow:
    """Generate a TwoPointCWindow object with 100 ells."""
    xy = TwoPointXY(x=harmonic_bin_1, y=harmonic_bin_2)
    two_point = TwoPointCWindow(XY=xy, window=window_1)
    return two_point


@pytest.fixture(name="wl_factory")
def make_wl_factory():
    """Generate a WeakLensingFactory object."""
    return wl.WeakLensingFactory(per_bin_systematics=[], global_systematics=[])


@pytest.fixture(name="nc_factory")
def make_nc_factory():
    """Generate a NumberCountsFactory object."""
    return nc.NumberCountsFactory(per_bin_systematics=[], global_systematics=[])


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
        measurement=Galaxies.COUNTS,
    )
    assert z_dist.bin_name == "bname1"
    assert z_dist.z[0] == 0
    assert z_dist.z[-1] == 1
    assert z_dist.dndz[0] == 1
    assert z_dist.dndz[-1] == 1
    assert z_dist.measurement == Galaxies.COUNTS


def test_inferred_galaxy_z_dist_bad_shape():
    with pytest.raises(
        ValueError, match="The z and dndz arrays should have the same shape."
    ):
        InferredGalaxyZDist(
            bin_name="bname1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(101),
            measurement=Clusters.COUNTS,
        )


def test_inferred_galaxy_z_dist_bad_type():
    with pytest.raises(ValueError, match="The measurement should be a Measurement."):
        InferredGalaxyZDist(
            bin_name="bname1",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurement=0,  # type: ignore
        )


def test_inferred_galaxy_z_dist_bad_name():
    with pytest.raises(ValueError, match="The bin_name should not be empty."):
        InferredGalaxyZDist(
            bin_name="",
            z=np.linspace(0, 1, 100),
            dndz=np.ones(100),
            measurement=Galaxies.COUNTS,
        )


def test_two_point_xy_gal_gal():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    assert xy.x == x
    assert xy.y == y


def test_two_point_xy_cmb_gal():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=CMB.CONVERGENCE,
    )
    xy = TwoPointXY(x=x, y=y)
    assert xy.x == x
    assert xy.y == y


def test_two_point_xy_invalid():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.SHEAR_E,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.SHEAR_T,
    )
    with pytest.raises(
        ValueError,
        match=("Measurements .* and .* are not compatible."),
    ):
        TwoPointXY(x=x, y=y)


def test_two_point_cells():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    cells = TwoPointCells(ells=ells, XY=xy)

    assert_array_equal(cells.ells, ells)
    assert cells.XY == xy
    assert cells.get_sacc_name() == harmonic(x.measurement, y.measurement)


def test_two_point_cells_invalid_ells():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
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
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.SHEAR_T,
    )
    xy = TwoPointXY(x=x, y=y)
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
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    two_point = TwoPointCWindow(XY=xy, window=window)

    assert two_point.window == window
    assert two_point.XY == xy
    assert two_point.get_sacc_name() == harmonic(x.measurement, y.measurement)


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
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.SHEAR_T,
    )
    xy = TwoPointXY(x=x, y=y)
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
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.SHEAR_T,
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
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    theta = np.array(np.linspace(0, 100, 100))
    two_point = TwoPointXiTheta(XY=xy, thetas=theta)

    assert_array_equal(two_point.thetas, theta)
    assert two_point.XY == xy
    assert two_point.get_sacc_name() == real(x.measurement, y.measurement)


def test_two_point_xi_theta_invalid():
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.SHEAR_E,
    )
    xy = TwoPointXY(x=x, y=y)
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


def test_two_point_xy_serialization(
    harmonic_bin_1: InferredGalaxyZDist, harmonic_bin_2: InferredGalaxyZDist
):
    xy = TwoPointXY(x=harmonic_bin_1, y=harmonic_bin_2)
    s = xy.to_yaml()
    # Take a look at how hideous the generated string
    # is.
    recovered = TwoPointXY.from_yaml(s)
    assert xy == recovered


def test_two_point_cells_serialization(
    harmonic_bin_1: InferredGalaxyZDist, harmonic_bin_2: InferredGalaxyZDist
):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    xy = TwoPointXY(x=harmonic_bin_1, y=harmonic_bin_2)
    cells = TwoPointCells(ells=ells, XY=xy)
    s = cells.to_yaml()
    recovered = TwoPointCells.from_yaml(s)
    assert cells == recovered


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
    xy = TwoPointXY(x=real_bin_1, y=real_bin_2)
    theta = np.array(np.linspace(0, 10, 10))
    xi_theta = TwoPointXiTheta(XY=xy, thetas=theta)
    s = xi_theta.to_yaml()
    recovered = TwoPointXiTheta.from_yaml(s)
    assert xi_theta == recovered


def test_two_point_from_metadata_cells(
    harmonic_bin_1, harmonic_bin_2, wl_factory, nc_factory
):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    xy = TwoPointXY(x=harmonic_bin_1, y=harmonic_bin_2)
    cells = TwoPointCells(ells=ells, XY=xy)
    two_point = TwoPoint.from_metadata_cells([cells], wl_factory, nc_factory).pop()

    assert two_point is not None
    assert isinstance(two_point, TwoPoint)
    assert two_point.sacc_data_type == cells.get_sacc_name()

    assert isinstance(two_point.source0, SourceGalaxy)
    assert isinstance(two_point.source1, SourceGalaxy)

    assert_array_equal(two_point.source0.tracer_args.z, harmonic_bin_1.z)
    assert_array_equal(two_point.source0.tracer_args.z, harmonic_bin_1.z)

    assert_array_equal(two_point.source0.tracer_args.dndz, harmonic_bin_1.dndz)
    assert_array_equal(two_point.source1.tracer_args.dndz, harmonic_bin_2.dndz)


def test_two_point_from_metadata_cells_unsupported_type(wl_factory, nc_factory):
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    x = InferredGalaxyZDist(
        bin_name="bname1",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=CMB.CONVERGENCE,
    )
    y = InferredGalaxyZDist(
        bin_name="bname2",
        z=np.linspace(0, 1, 100),
        dndz=np.ones(100),
        measurement=Galaxies.COUNTS,
    )
    xy = TwoPointXY(x=x, y=y)
    cells = TwoPointCells(ells=ells, XY=xy)
    with pytest.raises(
        ValueError,
        match="Measurement .* not supported!",
    ):
        TwoPoint.from_metadata_cells([cells], wl_factory, nc_factory)
