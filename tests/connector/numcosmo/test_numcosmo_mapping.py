"""Unit tests for the numcosmo Mapping connector."""

import pytest
import pyccl as ccl
from numcosmo_py import Ncm, Nc

from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.connector.numcosmo.numcosmo import (
    NumCosmoData,
    NumCosmoGaussCov,
    MappingNumCosmo,
    NumCosmoFactory,
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
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        RuntimeError,
        match="Model name non_existing_model was not found in the model set.",
    ):
        map_cosmo.create_params_map(["non_existing_model"], mset)


def test_numcosmo_mapping_create_params_map_absent_model():
    """Test the NumCosmo mapping connector create_params_map
    with an existing type but not present in the model set."""

    cosmo = Nc.HICosmoDEXcdm()

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
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
        map_cosmo.create_params_map(["MyModel"], mset)


def test_numcosmo_mapping_create_params_map_two_models_sharing_parameters():
    """Test the NumCosmo mapping connector create_params_map
    with an existing type but not present in the model set."""

    cosmo = Nc.HICosmoDEXcdm()

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
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
        map_cosmo.create_params_map(["MyModel1", "MyModel2"], mset)


def test_numcosmo_mapping_unsupported():
    """Test the NumCosmo mapping connector with an unsupported model."""

    cosmo = Nc.HICosmoDEJbp()

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
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
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    map_cosmo.set_params_from_numcosmo(mset)
    ccl_args = map_cosmo.calculate_ccl_args(mset)
    ccl_cosmo = ccl.CosmologyCalculator(**map_cosmo.mapping.asdict(), **ccl_args)

    assert ccl_cosmo["H0"] == cosmo.param_get_by_name("H0")
    assert ccl_cosmo["Omega_c"] == cosmo.param_get_by_name("Omegac")
    assert ccl_cosmo["Omega_b"] == cosmo.param_get_by_name("Omegab")
    assert ccl_cosmo["Omega_k"] == cosmo.param_get_by_name("Omegak")


@pytest.mark.parametrize(
    "numcosmo_cosmo_fixture",
    ["numcosmo_cosmo_xcdm", "numcosmo_cosmo_xcdm_no_nu", "numcosmo_cosmo_cpl"],
)
def test_numcosmo_serialize_mapping(numcosmo_cosmo_fixture, request):
    """Test the NumCosmo mapping connector consistence."""
    numcosmo_cosmo = request.getfixturevalue(numcosmo_cosmo_fixture)

    cosmo = numcosmo_cosmo["cosmo"]
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        p_ml=numcosmo_cosmo["p_ml"],
        p_mnl=numcosmo_cosmo["p_mnl"],
        dist=numcosmo_cosmo["dist"],
    )

    assert map_cosmo.mapping_name == "default"

    ser = Ncm.Serialize.new(Ncm.SerializeOpt.NONE)

    map_cosmo_dup = ser.dup_obj(map_cosmo)
    assert isinstance(map_cosmo_dup, MappingNumCosmo)
    assert map_cosmo_dup.mapping_name == "default"

    mset = Ncm.MSet()
    mset.set(cosmo)

    map_cosmo.set_params_from_numcosmo(mset)
    map_cosmo_dup.set_params_from_numcosmo(mset)

    if map_cosmo_dup.p_ml is None:
        assert map_cosmo_dup.p_ml is None
    else:
        assert id(map_cosmo_dup.p_ml) != id(map_cosmo.p_ml)
        assert isinstance(map_cosmo_dup.p_ml, Nc.PowspecML)

    if map_cosmo_dup.p_mnl is None:
        assert map_cosmo_dup.p_mnl is None
    else:
        assert id(map_cosmo_dup.p_mnl) != id(map_cosmo.p_mnl)
        assert isinstance(map_cosmo_dup.p_mnl, Nc.PowspecMNL)

    assert id(map_cosmo_dup.dist) != id(map_cosmo.dist)
    assert isinstance(map_cosmo_dup.dist, Nc.Distance)


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
    )

    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    fc_data = NumCosmoData.new_from_likelihood(
        likelihood=likelihood,
        tools=ModelingTools(),
        nc_mapping=map_cosmo,
        model_list=["NcFirecrownTrivial"],
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
    )

    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data_for_trivial_stat)

    fc_data = NumCosmoGaussCov.new_from_likelihood(
        likelihood=likelihood,
        tools=ModelingTools(),
        nc_mapping=map_cosmo,
        model_list=["NcFirecrownTrivial"],
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
@pytest.mark.parametrize(
    "likelihood_file",
    [
        "tests/likelihood/lkdir/lkscript.py",
        "tests/likelihood/gauss_family/lkscript_const_gaussian.py",
    ],
)
def test_numcosmo_serialize_likelihood(
    numcosmo_cosmo_fixture,
    likelihood_file,
    request,
):
    """Test the NumCosmo data connector for NcmData with serialization."""

    numcosmo_cosmo = request.getfixturevalue(numcosmo_cosmo_fixture)

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        p_ml=numcosmo_cosmo["p_ml"],
        p_mnl=numcosmo_cosmo["p_mnl"],
        dist=numcosmo_cosmo["dist"],
    )

    nc_factory = NumCosmoFactory(
        likelihood_file,
        NamedParameters(),
        map_cosmo,
        model_list=["NcFirecrownTrivial"],
    )

    fc_data = nc_factory.get_data()

    assert isinstance(fc_data, (NumCosmoData, NumCosmoGaussCov))
    assert fc_data.get_length() > 0
    assert fc_data.get_dof() > 0

    ser = Ncm.Serialize.new(Ncm.SerializeOpt.NONE)

    fc_data_dup = ser.dup_obj(fc_data)
    assert isinstance(fc_data_dup, (NumCosmoData, NumCosmoGaussCov))

    if fc_data.nc_mapping is None:
        assert fc_data_dup.nc_mapping is None
    else:
        assert id(fc_data_dup.nc_mapping) != id(fc_data.nc_mapping)
        assert isinstance(fc_data_dup.nc_mapping, MappingNumCosmo)

    assert id(fc_data_dup.likelihood) != id(fc_data.likelihood)
    assert isinstance(fc_data_dup.likelihood, Likelihood)

    assert id(fc_data_dup.tools) != id(fc_data.tools)
    assert isinstance(fc_data_dup.tools, ModelingTools)

    assert id(fc_data_dup.model_list) != id(fc_data.model_list)
    assert isinstance(fc_data_dup.model_list, list)
    assert fc_data_dup.model_list == fc_data.model_list
