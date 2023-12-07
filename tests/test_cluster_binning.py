"""Tests for the cluster binning module"""
import sacc
from firecrown.models.cluster.binning import SaccBin, NDimensionalBin
from unittest.mock import Mock


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
    sb = SaccBin([tracer_z, tracer_lambda])
    assert sb.z_edges == (0, 1)

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb = SaccBin([tracer_lambda, tracer_z])
    assert sb.z_edges == (0, 1)


def test_sacc_bin_richness_edges():
    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb = SaccBin([tracer_z, tracer_lambda])
    assert sb.mass_proxy_edges == (4, 5)

    tracer_z = sacc.tracers.BinZTracer("", 0, 1)
    tracer_lambda = sacc.tracers.BinRichnessTracer("", 4, 5)
    sb = SaccBin([tracer_lambda, tracer_z])
    assert sb.mass_proxy_edges == (4, 5)


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
