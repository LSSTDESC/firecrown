"""Tests for the cluster binning module"""
import sacc
from firecrown.models.cluster.binning import SaccBin


def test_create_sacc_bin_with_correct_dimension():
    tracer = sacc.tracers.SurveyTracer("", 1)
    for i in range(10):
        tracers = [tracer for _ in range(i)]
        sb = SaccBin(tracers)
        assert sb is not None
        assert sb.dimension == i
