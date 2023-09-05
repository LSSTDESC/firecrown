"""Tests for the Supernova statistic.
"""
import pytest

import sacc

from firecrown.likelihood.gauss_family.statistic.supernova import Supernova


@pytest.fixture(name="minimal_stat")
def fixture_minimal_stat() -> Supernova:
    """Return a correctly initialized :python:`Supernova` object."""
    stat = Supernova(sacc_tracer="sn_fake_sample")
    return stat


@pytest.fixture(name="missing_sacc_tracer")
def fixture_missing_tracer() -> sacc.Sacc:
    """Return a sacc.Sacc object that lacks a sacc_tracer."""
    return sacc.Sacc()


@pytest.fixture(name="wrong_tracer_type")
def fixture_wrong_tracer_type() -> sacc.Sacc:
    data = sacc.Sacc()
    data.add_tracer("NZ", "sn_fake_sample", 1.0, 5.0)
    return data


@pytest.fixture(name="good_sacc_data")
def fixture_sacc_data() -> sacc.Sacc:
    """Return a sacc.Sacc object sufficient to correctly read a
    :python:`Supernova` object.
    """
    data = sacc.Sacc()
    data.add_tracer("Misc", "sn_fake_sample")
    # The value of 16.95 supplied here is what will be recovered as entry 0
    # of the statistic's data vector.
    data.add_data_point("supernova_distance_mu", ("sn_fake_sample",), 16.95, z=0.0413)
    # TODO: fill in the right stuff.

    return data


def test_missing_sacc_tracer_fails_read(
    minimal_stat: Supernova, missing_sacc_tracer: sacc.Sacc
):
    with pytest.raises(
        ValueError,
        match="The SACC file does not contain the MiscTracer sn_fake_sample",
    ):
        minimal_stat.read(missing_sacc_tracer)


def test_wrong_tracer_type_fails_read(
    minimal_stat: Supernova, wrong_tracer_type: sacc.Sacc
):
    with pytest.raises(
        ValueError,
        match=f"The SACC tracer {minimal_stat.sacc_tracer} is not a MiscTracer",
    ):
        minimal_stat.read(wrong_tracer_type)


def test_read_works(minimal_stat: Supernova, good_sacc_data: sacc.Sacc):
    """After read() is called, we should be able to get the statistic's

    :python:`DataVector` and also should be able to call
    :python:`compute_theory_vector`.
    """
    minimal_stat.read(good_sacc_data)
    data_vector = minimal_stat.get_data_vector()
    assert len(data_vector) == 1
    assert data_vector[0] == 16.95
