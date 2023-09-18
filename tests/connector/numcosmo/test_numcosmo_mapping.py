"""Unit tests for the numcosmo Mapping connector."""

import pytest
import pyccl as ccl
from numcosmo_py import Ncm, Nc

from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.connector.numcosmo.numcosmo import (
    NumCosmoData,
    NumCosmoGaussCov,
    MappingNumCosmo,
)

from firecrown.connector.numcosmo.model import (
    NumCosmoModel,
    ScalarParameter,
    define_numcosmo_model,
)

Ncm.cfg_init()


def test_numcosmo_mapping_create_params_map_non_existing_model():
    """Test the NumCosmo mapping connector create_params_map
    with an non existing type."""

    cosmo = Nc.HICosmoDEXcdm()

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
        model_list=["non_existing_model"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        RuntimeError,
        match="Model name non_existing_model was not found in the model set.",
    ):
        map_cosmo.create_params_map(mset)


def test_numcosmo_mapping_create_params_map_absent_model():
    """Test the NumCosmo mapping connector create_params_map
    with an existing type but not present in the model set."""

    cosmo = Nc.HICosmoDEXcdm()

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
        model_list=["MyModel"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    my_model_dc = NumCosmoModel(
        name="MyModel", description="MyModel desc", scalar_params=[], vector_params=[]
    )

    MyModel = define_numcosmo_model(my_model_dc)

    my_model = MyModel()
    del my_model

    with pytest.raises(
        RuntimeError,
        match="Model MyModel was not found in the model set.",
    ):
        map_cosmo.create_params_map(mset)


def test_numcosmo_mapping_create_params_map_two_models_sharing_parameters():
    """Test the NumCosmo mapping connector create_params_map
    with an existing type but not present in the model set."""

    cosmo = Nc.HICosmoDEXcdm()

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
        model_list=["MyModel1", "MyModel2"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    my_model1_dc = NumCosmoModel(
        name="MyModel1",
        description="MyModel1 desc",
        scalar_params=[
            ScalarParameter(symbol="symbol1", name="param1", default_value=1.0),
            ScalarParameter(symbol="symbol2", name="param2", default_value=1.0),
        ],
        vector_params=[],
    )
    my_model2_dc = NumCosmoModel(
        name="MyModel2",
        description="MyModel2 desc",
        scalar_params=[
            ScalarParameter(symbol="symbol2", name="param2", default_value=1.0),
        ],
        vector_params=[],
    )

    MyModel1 = define_numcosmo_model(my_model1_dc)
    MyModel2 = define_numcosmo_model(my_model2_dc)

    my_model1 = MyModel1()
    my_model2 = MyModel2()

    mset.set(my_model1)
    mset.set(my_model2)

    with pytest.raises(
        RuntimeError,
        match="The following keys .* appear in more than one model used by the module",
    ):
        map_cosmo.create_params_map(mset)


def test_numcosmo_mapping_unsupported():
    """Test the NumCosmo mapping connector with an unsupported model."""

    cosmo = Nc.HICosmoDEJbp()

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
        model_list=["non_existing_model"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(ValueError, match="NumCosmo object .* not supported."):
        map_cosmo.set_params_from_numcosmo(mset)


def test_numcosmo_mapping_missing_hiprim():
    """Test the NumCosmo mapping connector with a model missing hiprim."""

    cosmo = Nc.HICosmoDECpl()

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
        model_list=["non_existing_model"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        ValueError, match="NumCosmo object must include a HIPrim object."
    ):
        map_cosmo.set_params_from_numcosmo(mset)


def test_numcosmo_mapping_invalid_hiprim():
    """Test the NumCosmo mapping connector with a model an invalid hiprim."""

    cosmo = Nc.HICosmoDECpl()
    prim = Nc.HIPrimAtan()
    cosmo.add_submodel(prim)

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
        model_list=["non_existing_model"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        ValueError, match="NumCosmo HIPrim object type .* not supported."
    ):
        map_cosmo.set_params_from_numcosmo(mset)


def test_numcosmo_mapping_no_p_mnl_require_nonlinear_pk():
    """Test the NumCosmo mapping connector with a model without p_mnl but
    with require_nonlinear_pk=True."""

    cosmo = Nc.HICosmoDECpl()
    prim = Nc.HIPrimPowerLaw()
    cosmo.add_submodel(prim)

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
        model_list=["non_existing_model"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    map_cosmo.set_params_from_numcosmo(mset)

    ccl_args = map_cosmo.calculate_ccl_args(mset)

    assert ccl_args["nonlinear_model"] == "halofit"


def test_numcosmo_mapping_no_p_mnl():
    """Test the NumCosmo mapping connector with a model without p_mnl and
    require_nonlinear_pk=False."""

    cosmo = Nc.HICosmoDECpl()
    prim = Nc.HIPrimPowerLaw()
    cosmo.add_submodel(prim)

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=False,
        dist=Nc.Distance.new(6.0),
        model_list=["non_existing_model"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    map_cosmo.set_params_from_numcosmo(mset)

    ccl_args = map_cosmo.calculate_ccl_args(mset)

    assert ccl_args["nonlinear_model"] is None


@pytest.mark.parametrize(
    "numcosmo_cosmo_fixture",
    ["numcosmo_cosmo_xcdm", "numcosmo_cosmo_xcdm_no_nu", "numcosmo_cosmo_cpl"],
)
def test_numcosmo_mapping(numcosmo_cosmo_fixture, request):
    """Test the NumCosmo mapping connector consistence."""
    numcosmo_cosmo = request.getfixturevalue(numcosmo_cosmo_fixture)

    cosmo = numcosmo_cosmo["cosmo"]
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        p_ml=numcosmo_cosmo["p_ml"],
        p_mnl=numcosmo_cosmo["p_mnl"],
        dist=numcosmo_cosmo["dist"],
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


@pytest.mark.parametrize(
    "numcosmo_cosmo_fixture",
    ["numcosmo_cosmo_xcdm", "numcosmo_cosmo_xcdm_no_nu", "numcosmo_cosmo_cpl"],
)
def test_numcosmo_data(
    numcosmo_cosmo_fixture,
    trivial_stats,
    sacc_data_for_trivial_stat,
    nc_model_trivial,
    request,
):
    """Test the NumCosmo data connector for NcmData."""

    numcosmo_cosmo = request.getfixturevalue(numcosmo_cosmo_fixture)

    cosmo = numcosmo_cosmo["cosmo"]
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        p_ml=numcosmo_cosmo["p_ml"],
        p_mnl=numcosmo_cosmo["p_mnl"],
        dist=numcosmo_cosmo["dist"],
        model_list=["NcFirecrownTrivial"],
    )

    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    fc_data = NumCosmoData(
        likelihood=likelihood,
        tools=ModelingTools(),
        mapping=map_cosmo,
    )

    assert fc_data.get_length() > 0
    assert fc_data.get_dof() > 0

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


@pytest.mark.parametrize(
    "numcosmo_cosmo_fixture",
    ["numcosmo_cosmo_xcdm", "numcosmo_cosmo_xcdm_no_nu", "numcosmo_cosmo_cpl"],
)
def test_numcosmo_gauss_cov(
    numcosmo_cosmo_fixture,
    trivial_stats,
    sacc_data_for_trivial_stat,
    nc_model_trivial,
    request,
):
    """Test the NumCosmo data connector for NcmDataGaussCov."""

    numcosmo_cosmo = request.getfixturevalue(numcosmo_cosmo_fixture)

    cosmo = numcosmo_cosmo["cosmo"]
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        p_ml=numcosmo_cosmo["p_ml"],
        p_mnl=numcosmo_cosmo["p_mnl"],
        dist=numcosmo_cosmo["dist"],
        model_list=["NcFirecrownTrivial"],
    )

    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    fc_data = NumCosmoGaussCov(
        likelihood=likelihood,
        tools=ModelingTools(),
        mapping=map_cosmo,
    )

    assert fc_data.get_length() > 0
    assert fc_data.get_dof() > 0

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
