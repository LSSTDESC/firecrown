"""Unit tests for the numcosmo Mapping connector."""

from typing import cast
import unittest.mock as mock
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
    helpers,
    create_params_map,
)
from firecrown.ccl_factory import PoweSpecAmplitudeParameter

Ncm.cfg_init()


@pytest.fixture(name="map_cosmo_dist")
def fixture_map_cosmo_dist() -> MappingNumCosmo:
    """Return a MappingNumCosmo with only distances."""
    return MappingNumCosmo(dist=Nc.Distance.new(6.0))


@pytest.fixture(name="map_cosmo_spec_nl")
def fixture_map_cosmo_spec_nl(numcosmo_cosmo) -> MappingNumCosmo:
    """Return a MappingNumCosmo instance with the specified p_ml, p_mnl, and dist."""
    return MappingNumCosmo(
        p_ml=numcosmo_cosmo["p_ml"],
        p_mnl=numcosmo_cosmo["p_mnl"],
        dist=numcosmo_cosmo["dist"],
    )


def test_create_params_map_non_existing_model(
    map_cosmo_dist: MappingNumCosmo,
):
    """Test the NumCosmo mapping connector create_params_map
    with an non existing type."""

    cosmo = Nc.HICosmoDEXcdm()

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        RuntimeError,
        match="Model name non_existing_model was not found in the model set.",
    ):
        create_params_map(["non_existing_model"], mset, map_cosmo_dist.mapping)


def test_numcosmo_mapping_create_params_map_absent_model(
    map_cosmo_dist: MappingNumCosmo,
):
    """Test the NumCosmo mapping connector create_params_map
    with an existing type but not present in the model set."""

    cosmo = Nc.HICosmoDEXcdm()

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        RuntimeError,
        match="Model name MyModel was not found in the model set.",
    ):
        create_params_map(["MyModel"], mset, map_cosmo_dist.mapping)


def test_numcosmo_mapping_create_params_map_two_models_sharing_parameters(
    map_cosmo_dist: MappingNumCosmo,
):
    """Test the NumCosmo mapping connector create_params_map
    with an existing type but not present in the model set."""

    cosmo = Nc.HICosmoDEXcdm()

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
        create_params_map(["MyModel1", "MyModel2"], mset, map_cosmo_dist.mapping)


def test_numcosmo_mapping_unsupported(map_cosmo_dist: MappingNumCosmo):
    """Test the NumCosmo mapping connector with an unsupported model."""

    cosmo = Nc.HICosmoDEJbp()

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(ValueError, match="NumCosmo object .* not supported."):
        map_cosmo_dist.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.SIGMA8)


def test_numcosmo_mapping_missing_hiprim(map_cosmo_dist: MappingNumCosmo):
    """Test the NumCosmo mapping connector with a model missing hiprim."""

    cosmo = Nc.HICosmoDECpl()

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        ValueError, match="NumCosmo object must include a HIPrim object."
    ):
        map_cosmo_dist.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.AS)


def test_numcosmo_mapping_invalid_hiprim(map_cosmo_dist: MappingNumCosmo):
    """Test the NumCosmo mapping connector with a model an invalid hiprim."""

    cosmo = Nc.HICosmoDECpl()
    prim = Nc.HIPrimAtan()
    cosmo.add_submodel(prim)

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        ValueError, match="NumCosmo HIPrim object type .* not supported."
    ):
        map_cosmo_dist.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.AS)


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

    map_cosmo.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.SIGMA8)
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

    map_cosmo.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.SIGMA8)
    map_cosmo_dup.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.SIGMA8)

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


def test_numcosmo_mapping_sigma8_missing_pk(map_cosmo_dist: MappingNumCosmo):
    """Test the MappingNumCosmo with sigma8 as a parameter but missing pk."""

    cosmo = Nc.HICosmoDEXcdm()

    mset = Ncm.MSet()
    mset.set(cosmo)

    with pytest.raises(
        ValueError, match="PowspecML object must be provided when using sigma8."
    ):
        map_cosmo_dist.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.SIGMA8)


@pytest.fixture(name="numcosmo_cosmo")
def fixture_numcosmo_cosmo():
    """Create a mock NumCosmo cosmology instance."""
    return mock.Mock(spec=Nc.HICosmo)


def test_get_amplitude_parameters_sigma8_no_powerspectrum(numcosmo_cosmo):
    """Test the get_amplitude_parameters function with sigma8 as a parameter but
    without a power spectrum."""

    with pytest.raises(
        ValueError, match="PowspecML object must be provided when using sigma8."
    ):
        _, _ = helpers.get_amplitude_parameters(
            PoweSpecAmplitudeParameter.SIGMA8, None, numcosmo_cosmo
        )


def test_accessors_with_no_powerspectrum():
    mapping = MappingNumCosmo()
    assert mapping.p_ml is None
    assert mapping.p_mnl is None
    mapping.p_ml = None
    assert mapping.p_ml is None
    mapping.p_mnl = None
    assert mapping.p_mnl is None

    linear = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
    nonlinear = Nc.PowspecMNLHaloFit.new(linear, 3.0, 1.0e-5)
    mapping.p_mnl = nonlinear
    assert mapping._p is None  # pylint: disable=protected-access


def test_accessors_with_only_linear_powerspectrum():
    linear = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
    mapping = MappingNumCosmo(p_ml=linear)
    assert mapping._p is not None  # pylint: disable=protected-access
    mapping.p_ml = linear
    assert mapping.p_mnl is None

    # Change to a different linear power spectrum
    other_linear = Nc.PowspecMLTransfer.new(Nc.TransferFuncBBKS.new())
    mapping.p_ml = other_linear
    assert mapping._p is not None  # pylint: disable=protected-access
    assert mapping.p_ml == other_linear
    assert mapping.p_mnl is None

    # Add a non-linear component
    nonlinear = Nc.PowspecMNLHaloFit.new(other_linear, 3.0, 1.0e-5)
    mapping.p_mnl = nonlinear
    assert mapping._p is not None  # pylint: disable=protected-access
    assert mapping.p_ml == other_linear
    assert mapping.p_mnl == nonlinear


def test_mapping_ccl_args_bg_only():
    """Test the MappingNumCosmo class with background only."""
    mapping = MappingNumCosmo(dist=Nc.Distance.new(6.0))
    assert mapping.p_ml is None
    assert mapping.p_mnl is None

    cosmo = Nc.HICosmoDEXcdm()
    cosmo.add_submodel(Nc.HIPrimPowerLaw.new())
    cosmo.add_submodel(Nc.HIReionCamb.new())

    mset = Ncm.MSet.new_array([cosmo])
    mset.prepare_fparam_map()

    mapping.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.AS)
    ccl_args = mapping.calculate_ccl_args(mset)
    assert "background" in ccl_args
    assert "pk_linear" not in ccl_args
    assert "pk_nonlin" not in ccl_args


def test_mapping_ccl_args_bg_pk_ml():
    """Test the MappingNumCosmo class with background and pk_ml."""
    mapping = MappingNumCosmo(
        dist=Nc.Distance.new(6.0),
        p_ml=Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new()),
    )
    assert mapping.p_mnl is None

    cosmo = Nc.HICosmoDEXcdm()
    cosmo.add_submodel(Nc.HIPrimPowerLaw.new())
    cosmo.add_submodel(Nc.HIReionCamb.new())

    mset = Ncm.MSet.new_array([cosmo])
    mset.prepare_fparam_map()

    mapping.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.AS)
    ccl_args = mapping.calculate_ccl_args(mset)
    assert "background" in ccl_args
    assert "pk_linear" in ccl_args
    assert "pk_nonlin" not in ccl_args


def test_mapping_ccl_args_bg_pk_ml_pk_mnl():
    """Test the MappingNumCosmo class with background, pk_ml and pk_mnl."""
    p_ml = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
    mapping = MappingNumCosmo(
        dist=Nc.Distance.new(6.0),
        p_ml=p_ml,
        p_mnl=Nc.PowspecMNLHaloFit.new(p_ml, 3.0, 1.0e-4),
    )
    assert mapping.p_ml is not None
    assert mapping.p_mnl is not None

    cosmo = Nc.HICosmoDEXcdm()
    cosmo.add_submodel(Nc.HIPrimPowerLaw.new())
    cosmo.add_submodel(Nc.HIReionCamb.new())

    mset = Ncm.MSet.new_array([cosmo])
    mset.prepare_fparam_map()

    mapping.set_params_from_numcosmo(mset, PoweSpecAmplitudeParameter.AS)
    ccl_args = mapping.calculate_ccl_args(mset)
    assert "background" in ccl_args
    assert "pk_linear" in ccl_args
    assert "pk_nonlin" in ccl_args


def test_mapping_ccl_args_bg_pk_mnl():
    """Test the MappingNumCosmo class with background, pk_ml and pk_mnl."""
    p_ml = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
    with pytest.raises(
        AssertionError, match="PowspecML object must be provided when using PowspecMNL"
    ):
        _ = MappingNumCosmo(
            dist=Nc.Distance.new(6.0),
            p_mnl=Nc.PowspecMNLHaloFit.new(p_ml, 3.0, 1.0e-4),
        )
