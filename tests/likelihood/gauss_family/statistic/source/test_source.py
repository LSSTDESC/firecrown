import pytest
import pyccl
from firecrown.likelihood.gauss_family.statistic.source.source import Tracer


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
