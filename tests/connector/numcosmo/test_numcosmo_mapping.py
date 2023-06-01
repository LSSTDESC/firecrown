"""Unit tests for the numcosmo Mapping connector."""

import pyccl as ccl
from numcosmo_py import Ncm

from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.connector.numcosmo.numcosmo import NumCosmoData, NumCosmoGaussCov

from firecrown.connector.numcosmo.numcosmo import MappingNumCosmo

Ncm.cfg_init()


def test_numcosmo_mapping(numcosmo_cosmo):
    """Test the NumCosmo mapping connector."""

    cosmo = numcosmo_cosmo["cosmo"]
    p_ml = numcosmo_cosmo["p_ml"]
    p_mnl = numcosmo_cosmo["p_mnl"]
    dist = numcosmo_cosmo["dist"]
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        p_ml=p_ml,
        p_mnl=p_mnl,
        dist=dist,
        model_list=["non_existing_model"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    map_cosmo.set_params_from_numcosmo(mset)
    ccl_args = map_cosmo.calculate_ccl_args(mset)
    ccl_cosmo = ccl.CosmologyCalculator(**map_cosmo.asdict(), **ccl_args)

    assert ccl_cosmo["H0"] == cosmo.param_get_by_name("H0")
    assert ccl_cosmo["Omega_c"] == cosmo.param_get_by_name("Omegac")
    assert ccl_cosmo["Omega_b"] == cosmo.param_get_by_name("Omegab")
    assert ccl_cosmo["Omega_k"] == cosmo.param_get_by_name("Omegak")


def test_numcosmo_data(numcosmo_cosmo, trivial_stats, sacc_data, nc_model_trivial):
    """Test the NumCosmo data connector for NcmData."""
    cosmo = numcosmo_cosmo["cosmo"]
    p_ml = numcosmo_cosmo["p_ml"]
    p_mnl = numcosmo_cosmo["p_mnl"]
    dist = numcosmo_cosmo["dist"]
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        p_ml=p_ml,
        p_mnl=p_mnl,
        dist=dist,
        model_list=["NcFirecrownTrivial"],
    )

    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data)

    fc_data = NumCosmoData(
        likelihood=likelihood,
        tools=ModelingTools(),
        mapping=map_cosmo,
    )

    dset = Ncm.Dataset()
    dset.append_data(fc_data)

    fc_trivial = nc_model_trivial()
    fc_trivial.param_set_by_name("mean", 1.0)

    mset = Ncm.MSet()
    mset.set(cosmo)
    mset.set(fc_trivial)
    mset.prepare_fparam_map()

    nc_likelihood = Ncm.Likelihood(dataset=dset)

    assert nc_likelihood.m2lnL_val(mset) == 2.0


def test_numcosmo_gauss_cov(numcosmo_cosmo, trivial_stats, sacc_data, nc_model_trivial):
    """Test the NumCosmo data connector for NcmDataGaussCov."""
    cosmo = numcosmo_cosmo["cosmo"]
    p_ml = numcosmo_cosmo["p_ml"]
    p_mnl = numcosmo_cosmo["p_mnl"]
    dist = numcosmo_cosmo["dist"]
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        p_ml=p_ml,
        p_mnl=p_mnl,
        dist=dist,
        model_list=["NcFirecrownTrivial"],
    )

    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data)

    fc_data = NumCosmoGaussCov(
        likelihood=likelihood,
        tools=ModelingTools(),
        mapping=map_cosmo,
    )

    dset = Ncm.Dataset()
    dset.append_data(fc_data)

    fc_trivial = nc_model_trivial()
    fc_trivial.param_set_by_name("mean", 1.0)

    mset = Ncm.MSet()
    mset.set(cosmo)
    mset.set(fc_trivial)
    mset.prepare_fparam_map()

    nc_likelihood = Ncm.Likelihood(dataset=dset)

    assert nc_likelihood.m2lnL_val(mset) == 2.0
