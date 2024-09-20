"""Unit tests for the numcosmo Mapping connector."""

from typing import cast
import pytest
import pyccl as ccl
from numcosmo_py import Ncm, Nc, GObject

from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.connector.numcosmo.numcosmo import (
    NumCosmoData,
    NumCosmoGaussCov,
    MappingNumCosmo,
    NumCosmoFactory,
)
from firecrown.ccl_factory import CCLFactory, PoweSpecAmplitudeParameter

Ncm.cfg_init()


def test_numcosmo_mapping_create_params_map_non_existing_model():
    """Test the NumCosmo mapping connector create_params_map
    with an non existing type."""

    cosmo = Nc.HICosmoDEXcdm()

    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))
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

    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))
    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        RuntimeError,
        match="Model name MyModel was not found in the model set.",
    ):
        map_cosmo.create_params_map(["MyModel"], mset)


def test_numcosmo_mapping_create_params_map_two_models_sharing_parameters():
    """Test the NumCosmo mapping connector create_params_map
    with an existing type but not present in the model set."""

    cosmo = Nc.HICosmoDEXcdm()
    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))

    mset = Ncm.MSet()
    mset.set(cosmo)

    my_model1_dc_yaml = r"""
NcmModelBuilder:
    parent-type-string: 'NcmModel'
    name: 'MyModel1'
    description: 'My Test Model 1'
    sparams:
    - NcmSParam:
        name: 'param1'
        symbol: 'symbol1'
        lower-bound: -5.0
        upper-bound: 5.0
        scale: 1.0
        absolute-tolerance: 0.0
        default-value: 1.0
        fit-type: 0
    - NcmSParam:
        name: 'param2'
        symbol: 'symbol2'
        lower-bound: -5.0
        upper-bound: 5.0
        scale: 1.0
        absolute-tolerance: 0.0
        default-value: 1.0
        fit-type: 0
"""

    my_model2_dc_yaml = r"""
NcmModelBuilder:
    parent-type-string: 'NcmModel'
    name: 'MyModel2'
    description: 'My Test Model 2'
    sparams:
    - NcmSParam:
        name: 'param2'
        symbol: 'symbol2'
        lower-bound: -5.0
        upper-bound: 5.0
        scale: 1.0
        absolute-tolerance: 0.0
        default-value: 1.0
        fit-type: 0
"""

    ser = Ncm.Serialize.new(Ncm.SerializeOpt.NONE)
    mb_model1: Ncm.ModelBuilder = cast(
        Ncm.ModelBuilder, ser.from_yaml(my_model1_dc_yaml)  # pylint: disable=no-member
    )
    assert isinstance(mb_model1, Ncm.ModelBuilder)
    model1_type = mb_model1.create()
    GObject.new(model1_type)

    mb_model2: Ncm.ModelBuilder = cast(
        Ncm.ModelBuilder, ser.from_yaml(my_model2_dc_yaml)  # pylint: disable=no-member
    )
    assert isinstance(mb_model2, Ncm.ModelBuilder)
    model2_type = mb_model2.create()
    GObject.new(model2_type)

    MyModel1 = model1_type.pytype
    MyModel2 = model2_type.pytype

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

    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))
    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(ValueError, match="NumCosmo object .* not supported."):
        map_cosmo.set_params_from_numcosmo(mset, CCLFactory())


def test_numcosmo_mapping_missing_hiprim():
    """Test the NumCosmo mapping connector with a model missing hiprim."""

    cosmo = Nc.HICosmoDECpl()

    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))
    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        ValueError, match="NumCosmo object must include a HIPrim object."
    ):
        map_cosmo.set_params_from_numcosmo(
            mset, CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.AS)
        )


def test_numcosmo_mapping_invalid_hiprim():
    """Test the NumCosmo mapping connector with a model an invalid hiprim."""

    cosmo = Nc.HICosmoDECpl()
    prim = Nc.HIPrimAtan()
    cosmo.add_submodel(prim)

    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))
    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        ValueError, match="NumCosmo HIPrim object type .* not supported."
    ):
        map_cosmo.set_params_from_numcosmo(
            mset, CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.AS)
        )


@pytest.mark.parametrize(
    "numcosmo_cosmo_fixture",
    ["numcosmo_cosmo_xcdm", "numcosmo_cosmo_xcdm_no_nu", "numcosmo_cosmo_cpl"],
)
def test_numcosmo_mapping(numcosmo_cosmo_fixture, request):
    """Test the NumCosmo mapping connector consistence."""
    numcosmo_cosmo = request.getfixturevalue(numcosmo_cosmo_fixture)

    cosmo = numcosmo_cosmo["cosmo"]
    map_cosmo = MappingNumCosmo(
        p_ml=numcosmo_cosmo["p_ml"],
        p_mnl=numcosmo_cosmo["p_mnl"],
        dist=numcosmo_cosmo["dist"],
    )

    mset = Ncm.MSet()
    mset.set(cosmo)

    map_cosmo.set_params_from_numcosmo(mset, CCLFactory())
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
        p_ml=numcosmo_cosmo["p_ml"],
        p_mnl=numcosmo_cosmo["p_mnl"],
        dist=numcosmo_cosmo["dist"],
    )

    assert map_cosmo.mapping_name == "default"

    ser = Ncm.Serialize.new(Ncm.SerializeOpt.NONE)

    map_cosmo_dup = ser.dup_obj(map_cosmo)  # pylint: disable=no-member
    assert isinstance(map_cosmo_dup, MappingNumCosmo)
    assert map_cosmo_dup.mapping_name == "default"

    mset = Ncm.MSet()
    mset.set(cosmo)

    map_cosmo.set_params_from_numcosmo(mset, CCLFactory())
    map_cosmo_dup.set_params_from_numcosmo(mset, CCLFactory())

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

    fc_data_dup = ser.dup_obj(fc_data)  # pylint: disable=no-member
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


def test_numcosmo_mapping_deprecated_require_nonlinear_pk():
    """Test the MappingNumCosmo deprecated require_nonlinear_pk."""

    with pytest.deprecated_call():
        _ = MappingNumCosmo(require_nonlinear_pk=True)


def test_numcosmo_mapping_sigma8_missing_pk():
    """Test the MappingNumCosmo with sigma8 as a parameter but missing pk."""

    cosmo = Nc.HICosmoDEXcdm()

    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))
    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        ValueError, match="PowspecML object must be provided when using sigma8."
    ):
        map_cosmo.set_params_from_numcosmo(
            mset, CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
        )
