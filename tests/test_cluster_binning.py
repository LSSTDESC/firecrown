"""Tests for the cluster binning module"""

from unittest.mock import Mock

import pytest
import sacc

from firecrown.models.cluster import NDimensionalBin, SaccBin, TupleBin


def test_bin_str():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_z, tracer_lambda, tracer_radius])
    assert str(sb) == "[(0, 1), (4, 5), (1, 2)]\n"


def test_create_sacc_bin_with_correct_dimension():
    tracer = sacc.tracers.SurveyTracer("", 1)
    for i in range(10):
        tracers = [tracer for _ in range(i)]
        sb = SaccBin(tracers)
        assert sb is not None
        assert sb.dimension == i


def test_sacc_bin_z_edges():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_z, tracer_lambda, tracer_radius])
    assert sb.z_edges == (0, 1)

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_radius, tracer_lambda, tracer_z])
    assert sb.z_edges == (0, 1)

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_lambda, tracer_z, tracer_radius])
    assert sb.z_edges == (0, 1)


def test_sacc_bin_z_edges_throws_when_multiple_z_bins():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_z, tracer_z, tracer_lambda, tracer_radius])
    with pytest.raises(ValueError, match="SaccBin must have exactly one z bin"):
        print(sb.z_edges)


def test_sacc_bin_mass_proxy_edges_throws_when_multiple_mass_proxy_bins():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_z, tracer_lambda, tracer_lambda, tracer_radius])
    with pytest.raises(ValueError, match="SaccBin must have exactly one richness bin"):
        print(sb.mass_proxy_edges)


def test_sacc_bin_radius_edges_throws_when_multiple_radius_bins():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_z, tracer_lambda, tracer_radius, tracer_radius])
    with pytest.raises(ValueError, match="SaccBin must have exactly one radius bin"):
        print(sb.radius_edges)


def test_sacc_bin_radius_center_throws_when_multiple_radius_bins():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_z, tracer_lambda, tracer_radius, tracer_radius])
    with pytest.raises(ValueError, match="SaccBin must have exactly one radius bin"):
        print(sb.radius_center)


def test_sacc_bin_richness_edges():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb = SaccBin([tracer_z, tracer_lambda])
    assert sb.mass_proxy_edges == (4, 5)

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb = SaccBin([tracer_lambda, tracer_z])
    assert sb.mass_proxy_edges == (4, 5)


def test_sacc_bin_radius_edges():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_z, tracer_lambda, tracer_radius])
    assert sb.radius_edges == (1, 2)

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_radius, tracer_lambda, tracer_z])
    assert sb.radius_edges == (1, 2)

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_lambda, tracer_radius, tracer_z])
    assert sb.radius_edges == (1, 2)


def test_sacc_bin_radius_center():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_z, tracer_lambda, tracer_radius])
    radius_center = sb.radius_center
    assert radius_center == 1.5

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_radius, tracer_lambda, tracer_z])
    radius_center = sb.radius_center
    assert radius_center == 1.5

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    tracer_radius = sacc.tracers.BinRadiusTracer("", 1, 2, 1.5)
    sb = SaccBin([tracer_lambda, tracer_radius, tracer_z])
    radius_center = sb.radius_center
    assert radius_center == 1.5


def test_equal_sacc_bins_are_equal():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb1 = SaccBin([tracer_z, tracer_lambda])
    sb2 = SaccBin([tracer_z, tracer_lambda])

    assert sb1 == sb2
    assert sb1 is not sb2

    sb1 = SaccBin([tracer_lambda, tracer_z])
    sb2 = SaccBin([tracer_z, tracer_lambda])

    assert sb1 == sb2
    assert sb1 is not sb2


def test_sacc_bin_different_edges_not_equal():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb1 = SaccBin([tracer_z, tracer_lambda])

    tracer_z = sacc.tracers.BinZTracer("", 0, 2)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb2 = SaccBin([tracer_z, tracer_lambda])

    assert sb1 != sb2

    tracer_z = sacc.tracers.BinZTracer("", -1, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb2 = SaccBin([tracer_z, tracer_lambda])

    assert sb1 != sb2

    tracer_z = sacc.tracers.BinZTracer("", -1, 2)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb2 = SaccBin([tracer_z, tracer_lambda])

    assert sb1 != sb2


def test_sacc_bin_different_dimensions_not_equal():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_z2 = sacc.tracers.BinZTracer("", 1, 2)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb1 = SaccBin([tracer_z, tracer_z2, tracer_lambda])
    sb2 = SaccBin([tracer_z, tracer_lambda])

    assert sb1 != sb2


def test_sacc_bin_must_be_equal_type():
    other_bin = Mock(spec=NDimensionalBin)

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb = SaccBin([tracer_z, tracer_lambda])

    assert sb != other_bin


def test_create_tuple_bin():
    tb = TupleBin([(1, 2), (3, 4), (1, 2, 1.5)])
    assert tb is not None
    assert tb.dimension == 3
    assert tb.mass_proxy_edges == (1, 2)
    assert tb.z_edges == (3, 4)
    assert tb.radius_edges == (1, 2)
    assert tb.radius_center == 1.5


def test_tuple_bins_are_equal():
    tb1 = TupleBin([(1, 2), (3, 4)])
    tb2 = TupleBin([(1, 2), (3, 4)])
    assert tb1 == tb2
    assert hash(tb1) == hash(tb2)
    assert tb1 is not tb2


def test_tuple_bin_neq_sacc_bin():
    tb = TupleBin([(1, 2), (3, 4)])
    tracer_z = sacc.tracers.BinZTracer("", 3, 4)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 1, 2)
    sb = SaccBin([tracer_z, tracer_lambda])
    assert tb != sb


def test_tuple_bin_different_dimensions_not_equal():
    tb1 = TupleBin([(1, 2), (3, 4), (5, 6)])
    tb2 = TupleBin([(1, 2), (3, 4)])
    assert tb1 != tb2

    tb2 = TupleBin([(1, 2, 3), (3, 4)])
    tb1 = TupleBin([(1, 2), (3, 4)])
    assert tb1 != tb2


def test_tuple_bin_different_bins_not_equal():
    tb1 = TupleBin([(1, 2), (3, 4)])
    tb2 = TupleBin([(1, 2), (3, 5)])
    assert tb1 != tb2

    tb1 = TupleBin([(0, 2), (3, 4)])
    tb2 = TupleBin([(1, 2), (3, 4)])
    assert tb1 != tb2

    tb1 = TupleBin([(0, 2), (0, 4)])
    tb2 = TupleBin([(1, 2), (3, 4)])
    assert tb1 != tb2

    tb1 = TupleBin([(3, 4), (1, 2)])
    tb2 = TupleBin([(1, 2), (3, 4)])
    assert tb1 != tb2
