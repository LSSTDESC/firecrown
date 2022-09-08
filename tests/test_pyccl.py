import pytest
import pyccl


# Both sets of cosmological parameters are silly, but they are sufficient to initialize
# a pyccl.Cosmology object.


@pytest.fixture
def cosmo_params_1():
    return {"Omega_c": 0.0, "Omega_b": 0.0, "h": 1.0, "A_s": 0.0, "n_s": 0.0}


@pytest.fixture
def cosmo_params_2():
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
    # Currently this test fails. This seems to demonstrate that caching based on the
    # value of a Cosmology does us no good. Instead, we should make sure we don't call
    # an expensive function passing the same Cosmology multiple times.
    x = pyccl.Cosmology(**cosmo_params_1)
    y = pyccl.Cosmology(**cosmo_params_1)
    assert x == y
    assert hash(x) == hash(y)
