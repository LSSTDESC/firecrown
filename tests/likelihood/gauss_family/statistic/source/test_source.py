"""
Tests for the module firecrown.likelihood.gauss_family.statistic.source.source.
"""
import pytest
import pyccl
from firecrown.likelihood.gauss_family.statistic.source.source import Tracer
import firecrown.likelihood.gauss_family.statistic.source.number_counts as nc
from firecrown.parameters import ParamsMap


class TrivialTracer(Tracer):
    """This is the most trivial possible subclass of Tracer."""


@pytest.fixture(name="empty_pyccl_tracer")
def fixture_empty_pyccl_tracer():
    return pyccl.Tracer()


def test_trivial_tracer_construction(empty_pyccl_tracer):
    trivial = TrivialTracer(empty_pyccl_tracer)
    assert trivial.ccl_tracer is empty_pyccl_tracer
    # If we have not assigned a name through the initializer, the name of the resulting
    # Tracer object is the class name of the :python:`pyccl.Tracer` that was used to
    # create it.
    assert trivial.tracer_name == "Tracer"
    # If we have not supplied a field, and have not supplied a tracer_name, then the
    # field is "delta_matter".
    assert trivial.field == "delta_matter"
    assert trivial.pt_tracer is None
    assert trivial.halo_profile is None
    assert trivial.halo_2pt is None
    assert not trivial.has_pt
    assert not trivial.has_hm


def test_tracer_construction_with_name(empty_pyccl_tracer):
    named = TrivialTracer(empty_pyccl_tracer, tracer_name="Fred")
    assert named.ccl_tracer is empty_pyccl_tracer
    assert named.tracer_name == "Fred"
    assert named.field == "Fred"
    assert named.pt_tracer is None
    assert named.halo_profile is None
    assert named.halo_2pt is None
    assert not named.has_pt
    assert not named.has_hm


def test_linear_bias_systematic():
    a = nc.LinearBiasSystematic("xxx")
    assert isinstance(a, nc.LinearBiasSystematic)
    assert a.parameter_prefix == "xxx"
    assert a.alphag is None
    assert a.alphaz is None
    assert a.z_piv is None
    assert not a.is_updated()

    a.update(ParamsMap({"xxx_alphag": 1.0, "xxx_alphaz": 2.0, "xxx_z_piv": 1.5}))
    assert a.is_updated()
    assert a.alphag == 1.0
    assert a.alphaz == 2.0
    assert a.z_piv == 1.5

    a.reset()
    assert not a.is_updated()
    assert a.parameter_prefix == "xxx"
    assert a.alphag is None
    assert a.alphaz is None
    assert a.z_piv is None


def test_weak_lensing_source():
    pass


def test_number_counts_source():
    pass
