"""Tests for the cluster abundance module."""

import numpy as np
import pyccl
import pytest
import clmm
from firecrown.models.cluster.DS_from_mass import DS_from_mass


@pytest.fixture(name="ds_from_mass")
def fixture_ds_from_mass():
    """Test fixture that represents a  cluster DS_from_mass class."""
    ds = DS_from_mass()
    return ds


def test_ds_from_mass_init(ds_from_mass: DS_from_mass):
    """Test the initiation of a cluster DS_from_mass class."""
    assert ds_from_mass is not None
    assert ds_from_mass.cosmo is None
    assert isinstance(ds_from_mass.moo, clmm.Modeling)
    assert isinstance(ds_from_mass.cosmo_clmm, clmm.Cosmology)
    assert isinstance(ds_from_mass.moo, clmm.Modeling)


def test_ds_from_mass_update_ingredients(ds_from_mass: DS_from_mass):
    """Test the update of a cluster DS_from_mass class."""
    cosmo = pyccl.CosmologyVanillaLCDM()
    clmm_cosmo = clmm.Cosmology()
    clmm_cosmo._init_from_cosmo(cosmo)

    ds_from_mass.update_ingredients(cosmo)
    assert ds_from_mass.cosmo is not None
    assert ds_from_mass.cosmo == cosmo
    assert ds_from_mass.cosmo_clmm is not None


def test_ds_from_mass_calculate_DS_from_Mass(ds_from_mass: DS_from_mass):
    """Test the profile calculation of a cluster DS_from_mass class."""
    cosmo = pyccl.CosmologyVanillaLCDM()
    clmm_cosmo = clmm.Cosmology()
    clmm_cosmo._init_from_cosmo(cosmo)
    moo = clmm.Modeling(massdef="mean", delta_mdef=200, halo_profile_model="nfw")
    moo.set_cosmo(clmm_cosmo)
    moo.set_concentration(4)
    mass = 14.0
    redshift = 1.0
    moo.set_mass(mass)
    radii = np.arange(0.1, 1.0, 0.1)
    profile = moo.eval_excess_surface_density(radii, redshift)

    ds_from_mass.update_ingredients(cosmo)

    assert (
        ds_from_mass.calculate_DS_from_Mass(mass, redshift, radii).all()
        == profile.all()
    )
