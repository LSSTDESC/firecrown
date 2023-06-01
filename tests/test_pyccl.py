"""
Test a few of the features of pyccl upon which we rely, but which might change in
future release of pyccl.
"""
import pytest
import pyccl


# Both sets of cosmological parameters are silly, but they are sufficient to initialize
# a pyccl.Cosmology object.


@pytest.fixture(name="cosmo_params_1")
def fixture_cosmo_params_1():
    return {"Omega_c": 0.0, "Omega_b": 0.0, "h": 1.0, "A_s": 0.0, "n_s": 0.0}


@pytest.fixture(name="cosmo_params_2")
def fixture_cosmo_params_2():
    return {"Omega_c": 0.0, "Omega_b": 0.0, "h": 1.0, "A_s": 0.0, "n_s": 0.25}


def test_alias_of_cosmology_hashes_equal(cosmo_params_1):
    x = pyccl.Cosmology(**cosmo_params_1)
    y = x
    assert x == y
    assert hash(x) == hash(y)


def test_unequal_cosmologies_hash_unequal(cosmo_params_1, cosmo_params_2):
    x = pyccl.Cosmology(**cosmo_params_1)
    y = pyccl.Cosmology(**cosmo_params_2)
    assert x != y
    assert hash(x) != hash(y)


def test_equal_cosmologies_hash_equal(cosmo_params_1):
    # This test verifies expected (but not really desired) behavior. Two
    # Cosmology variables only have equal hashes when they are in fact they
    # are aliases for the same object. They also test as equal only if they
    # are aliases for the same object.
    x = pyccl.Cosmology(**cosmo_params_1)
    y = pyccl.Cosmology(**cosmo_params_1)
    # This behavior will change in future versions of pyccl,
    # making this test always pass. TODO: update this test when
    # pyccl is updated.
    assert (x != y and hash(x) != hash(y)) or (x == y and hash(x) == hash(y))
