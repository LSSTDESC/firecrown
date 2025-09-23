"""Unit tests for the numcosmo connector."""

import pytest
from numcosmo_py import Ncm, Nc, GObject

from firecrown.connector.numcosmo.numcosmo import (
    NumCosmoFactory,
    MappingNumCosmo,
    NumCosmoData,
    NumCosmoGaussCov,
)

from firecrown.likelihood.likelihood import NamedParameters, Likelihood, load_likelihood
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.updatable import get_default_params

Ncm.cfg_init()


@pytest.fixture(name="factory_plain")
def fixture_factory_plain():
    """Create a NumCosmoFactory instance."""
    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))
    build_parameters = NamedParameters()
    return NumCosmoFactory(
        "tests/likelihood/lkdir/lkscript.py",
        build_parameters,
        map_cosmo,
        model_list=["non_existing_model"],
    )


@pytest.fixture(name="factory_const_gauss")
def fixture_factory_const_gauss():
    """Create a NumCosmoFactory instance."""
    map_cosmo = MappingNumCosmo(dist=Nc.Distance.new(6.0))
    build_parameters = NamedParameters()
    return NumCosmoFactory(
        "tests/likelihood/gauss_family/lkscript_const_gaussian.py",
        build_parameters,
        map_cosmo,
        model_list=["non_existing_model"],
    )


def test_wrong_use_of_power_spectra_with_ccl_not_in_default_mode():
    map_cosmo = MappingNumCosmo(
        dist=Nc.Distance.new(6.0),
        p_ml=Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new()),
        p_mnl=Nc.PowspecMNLHaloFit.new(
            Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new()), 3.0, 1.0e-4
        ),
    )
    build_parameters = NamedParameters({"projection": "harmonic"})
    with pytest.raises(RuntimeError):
        NumCosmoFactory(
            "tests/likelihood/gauss_family/lkscript_two_point_pure_ccl.py",
            build_parameters,
            map_cosmo,
            model_list=["non_existing_model"],
        )


def test_numcosmo_connector_plain(factory_plain):
    """Test the NumCosmo connector."""

    assert factory_plain.get_data() is not None
    assert factory_plain.get_mapping() is not None


def test_numcosmo_connector_const_gauss(factory_const_gauss):
    """Test the NumCosmo connector."""

    assert factory_const_gauss.get_data() is not None
    assert factory_const_gauss.get_mapping() is not None


def test_numcosmo_factory_plain_get_firecrown_likelihood(factory_plain):
    """Test the NumCosmo connector."""

    assert factory_plain.get_data() is not None
    assert factory_plain.get_mapping() is not None

    fc_likelihood = factory_plain.get_firecrown_likelihood()
    assert fc_likelihood is not None
    assert isinstance(fc_likelihood, Likelihood)


def test_numcosmo_factory_const_gauss_get_firecrown_likelihood(factory_const_gauss):
    """Test the NumCosmo connector."""

    assert factory_const_gauss.get_data() is not None
    assert factory_const_gauss.get_mapping() is not None

    fc_likelihood = factory_const_gauss.get_firecrown_likelihood()
    assert fc_likelihood is not None
    assert isinstance(fc_likelihood, Likelihood)


def test_numcosmo_factory_plain_get_properties(factory_plain):
    """Test the NumCosmo connector."""

    nc_data = factory_plain.get_data()

    assert nc_data is not None
    assert isinstance(nc_data, Ncm.Data)
    assert isinstance(nc_data, NumCosmoData)

    assert nc_data.likelihood_source == "tests/likelihood/lkdir/lkscript.py"
    assert isinstance(nc_data.likelihood_build_parameters, Ncm.VarDict)


def test_numcosmo_factory_const_gauss_get_properties(factory_const_gauss):
    """Test the NumCosmo connector."""

    nc_data = factory_const_gauss.get_data()

    assert nc_data is not None
    assert isinstance(nc_data, Ncm.Data)
    assert isinstance(nc_data, NumCosmoGaussCov)

    assert (
        nc_data.likelihood_source
        == "tests/likelihood/gauss_family/lkscript_const_gaussian.py"
    )
    assert isinstance(nc_data.likelihood_build_parameters, Ncm.VarDict)


def test_numcosmo_factory_plain_set_properties_order1(factory_plain):
    """Test the NumCosmo connector."""

    nc_data = factory_plain.get_data()

    assert nc_data is not None
    assert isinstance(nc_data, Ncm.Data)
    assert isinstance(nc_data, NumCosmoData)

    old_fc_likelihood = nc_data.likelihood
    assert old_fc_likelihood is not None
    assert isinstance(old_fc_likelihood, Likelihood)

    nc_data.likelihood_source = "tests/likelihood/lkdir/lkscript.py"
    nc_data.likelihood_build_parameters = Ncm.VarDict.new()

    fc_likelihood = nc_data.likelihood
    assert fc_likelihood is not None
    assert isinstance(fc_likelihood, Likelihood)
    assert fc_likelihood != old_fc_likelihood


def test_numcosmo_factory_plain_set_properties_order2(factory_plain):
    """Test the NumCosmo connector."""

    nc_data = factory_plain.get_data()

    assert nc_data is not None
    assert isinstance(nc_data, Ncm.Data)
    assert isinstance(nc_data, NumCosmoData)

    old_fc_likelihood = nc_data.likelihood
    assert old_fc_likelihood is not None
    assert isinstance(old_fc_likelihood, Likelihood)

    nc_data.likelihood_build_parameters = Ncm.VarDict.new()
    nc_data.likelihood_source = "tests/likelihood/lkdir/lkscript.py"

    fc_likelihood = nc_data.likelihood
    assert fc_likelihood is not None
    assert isinstance(fc_likelihood, Likelihood)
    assert fc_likelihood != old_fc_likelihood


def test_numcosmo_factory_plain_set_properties_empty_build_parameters(factory_plain):
    """Test the NumCosmo connector."""

    nc_data = factory_plain.get_data()

    assert nc_data is not None
    assert isinstance(nc_data, Ncm.Data)
    assert isinstance(nc_data, NumCosmoData)

    old_fc_likelihood = nc_data.likelihood
    assert old_fc_likelihood is not None
    assert isinstance(old_fc_likelihood, Likelihood)

    nc_data.likelihood_source = "tests/likelihood/lkdir/lkscript.py"
    nc_data.likelihood_build_parameters = None

    fc_likelihood = nc_data.likelihood
    assert fc_likelihood is not None
    assert isinstance(fc_likelihood, Likelihood)
    assert fc_likelihood != old_fc_likelihood


def test_numcosmo_factory_const_gauss_set_properties_order1(factory_const_gauss):
    """Test the NumCosmo connector."""

    nc_data = factory_const_gauss.get_data()

    assert nc_data is not None
    assert isinstance(nc_data, Ncm.Data)
    assert isinstance(nc_data, NumCosmoGaussCov)

    old_fc_likelihood = nc_data.likelihood
    assert old_fc_likelihood is not None
    assert isinstance(old_fc_likelihood, Likelihood)

    nc_data.likelihood_source = (
        "tests/likelihood/gauss_family/lkscript_const_gaussian.py"
    )
    nc_data.likelihood_build_parameters = Ncm.VarDict.new()

    fc_likelihood = nc_data.likelihood
    assert fc_likelihood is not None
    assert isinstance(fc_likelihood, Likelihood)
    assert fc_likelihood != old_fc_likelihood


def test_numcosmo_factory_const_gauss_set_properties_order2(factory_const_gauss):
    """Test the NumCosmo connector."""

    nc_data = factory_const_gauss.get_data()

    assert nc_data is not None
    assert isinstance(nc_data, Ncm.Data)
    assert isinstance(nc_data, NumCosmoGaussCov)

    old_fc_likelihood = nc_data.likelihood
    assert old_fc_likelihood is not None
    assert isinstance(old_fc_likelihood, Likelihood)

    nc_data.likelihood_build_parameters = Ncm.VarDict.new()
    nc_data.likelihood_source = (
        "tests/likelihood/gauss_family/lkscript_const_gaussian.py"
    )

    fc_likelihood = nc_data.likelihood
    assert fc_likelihood is not None
    assert isinstance(fc_likelihood, Likelihood)
    assert fc_likelihood != old_fc_likelihood


def test_numcosmo_factory_const_gauss_set_properties_empty_build_parameters(
    factory_const_gauss,
):
    """Test the NumCosmo connector."""

    nc_data = factory_const_gauss.get_data()

    assert nc_data is not None
    assert isinstance(nc_data, Ncm.Data)
    assert isinstance(nc_data, NumCosmoGaussCov)

    old_fc_likelihood = nc_data.likelihood
    assert old_fc_likelihood is not None
    assert isinstance(old_fc_likelihood, Likelihood)

    nc_data.likelihood_source = (
        "tests/likelihood/gauss_family/lkscript_const_gaussian.py"
    )
    nc_data.likelihood_build_parameters = None

    fc_likelihood = nc_data.likelihood
    assert fc_likelihood is not None
    assert isinstance(fc_likelihood, Likelihood)
    assert fc_likelihood != old_fc_likelihood


def test_empty_plain_data():
    """Test the NumCosmo connector."""

    nc_data = NumCosmoData()
    assert nc_data.likelihood_source is None
    assert nc_data.likelihood_build_parameters is None


def test_empty_gauss_cov_data():
    """Test the NumCosmo connector."""

    nc_data = NumCosmoGaussCov()
    assert nc_data.likelihood_source is None
    assert nc_data.likelihood_build_parameters is None


@pytest.mark.slow
def test_default_factory_const_gauss():
    """Test the NumCosmo connector."""
    build_parameters = NamedParameters(
        {"likelihood_config": "examples/des_y1_3x2pt/pure_ccl_experiment.yaml"}
    )
    model_name = "firecrown_model_gauss"

    likelihood_source = "firecrown.likelihood.factories.build_two_point_likelihood"
    likelihood, tools = load_likelihood(likelihood_source, build_parameters)
    assert isinstance(likelihood, ConstGaussian)
    data = NumCosmoGaussCov.new_from_likelihood(
        likelihood,
        [model_name],
        tools,
        None,
        likelihood_source,
        build_parameters,
    )

    run_likelihood(model_name, data)


@pytest.mark.slow
def test_default_factory_plain():
    """Test the NumCosmo connector."""
    build_parameters = NamedParameters(
        {"likelihood_config": "examples/des_y1_3x2pt/pure_ccl_experiment.yaml"}
    )
    model_name = "firecrown_model_plain"

    likelihood_source = "firecrown.likelihood.factories.build_two_point_likelihood"
    likelihood, tools = load_likelihood(likelihood_source, build_parameters)
    assert isinstance(likelihood, ConstGaussian)
    data = NumCosmoData.new_from_likelihood(
        likelihood,
        [model_name],
        tools,
        None,
        likelihood_source,
        build_parameters,
    )

    run_likelihood(model_name, data)


def run_likelihood(model_name, data):
    """Run the likelihood."""
    default_parameters = get_default_params(data.likelihood, data.tools)
    model_builder = Ncm.ModelBuilder.new(
        Ncm.Model,
        model_name,
        f"Test model {model_name}",
    )
    for param, value in default_parameters.items():
        if isinstance(value, float):
            model_builder.add_sparam(
                param,
                param,
                -1.0e10,
                1.0e10,
                1.0e-2,
                0.0,
                value,
                Ncm.ParamType.FIXED,
            )
        else:
            assert isinstance(value, list)
            assert len(value) <= 1
            model_builder.add_sparam(
                param,
                param,
                -1.0e10,
                1.0e10,
                1.0e-2,
                0.0,
                value[0] if len(value) == 1 else 0.0,
                Ncm.ParamType.FIXED,
            )

    FirecrownModel = model_builder.create()  # pylint: disable=invalid-name
    GObject.new(FirecrownModel)
    NcmFirecrownModel = FirecrownModel.pytype  # pylint: disable=invalid-name
    GObject.type_register(NcmFirecrownModel)

    model: Ncm.Model = NcmFirecrownModel()
    model.params_set_default_ftype()

    assert data is not None
    assert isinstance(data, Ncm.Data)
    cosmo = Nc.HICosmoDEXcdm()
    cosmo.add_submodel(Nc.HIPrimPowerLaw.new())
    cosmo.add_submodel(Nc.HIReionCamb.new())
    mset = Ncm.MSet()
    mset.set(cosmo)
    mset.set(model)
    mset.prepare_fparam_map()
    data.prepare(mset)


def test_create_params_map_with_mapping():
    """Test create_params_map function when mapping is provided."""
    from firecrown.connector.numcosmo.numcosmo import create_params_map
    from firecrown.connector.mapping import Mapping
    from firecrown.parameters import ParamsMap

    # Create a simple mock model and mset
    model_name = "test_model"
    model_builder = Ncm.ModelBuilder.new(
        Ncm.Model,
        model_name,
        f"Test model {model_name}",
    )
    # Add a simple parameter
    model_builder.add_sparam(
        "test_param",
        "test_param",
        -1.0,
        1.0,
        1.0e-2,
        0.0,
        0.5,
        Ncm.ParamType.FIXED,
    )

    TestModel = model_builder.create()  # pylint: disable=invalid-name
    GObject.new(TestModel)
    NcmTestModel = TestModel.pytype  # pylint: disable=invalid-name
    GObject.type_register(NcmTestModel)

    model: Ncm.Model = NcmTestModel()
    model.params_set_default_ftype()

    mset = Ncm.MSet()
    mset.set(model)

    # Create a mapping with some parameters
    mapping = Mapping()
    mapping.set_params(
        Omega_c=0.26,
        Omega_b=0.048,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        Omega_k=0.0,
        Neff=3.046,
        m_nu=0.06,
        w0=-1.0,
        wa=0.0,
        T_CMB=2.725,
    )

    # Test create_params_map with mapping (should cover lines 342-345)
    params_map = create_params_map([model_name], mset, mapping)

    # Verify the function worked correctly
    assert isinstance(params_map, ParamsMap)
    # Should contain both model parameters and mapping parameters
    assert "test_param" in params_map
    assert "Omega_c" in params_map
    assert "h" in params_map
    assert params_map["Omega_c"] == 0.26
    assert params_map["h"] == 0.7


def test_create_params_map_without_mapping():
    """Test create_params_map function when mapping is None."""
    from firecrown.connector.numcosmo.numcosmo import create_params_map
    from firecrown.parameters import ParamsMap

    # Create a simple mock model and mset
    model_name = "test_model2"
    model_builder = Ncm.ModelBuilder.new(
        Ncm.Model,
        model_name,
        f"Test model {model_name}",
    )
    # Add a simple parameter
    model_builder.add_sparam(
        "test_param2",
        "test_param2",
        -1.0,
        1.0,
        1.0e-2,
        0.0,
        0.3,
        Ncm.ParamType.FIXED,
    )

    TestModel2 = model_builder.create()  # pylint: disable=invalid-name
    GObject.new(TestModel2)
    NcmTestModel2 = TestModel2.pytype  # pylint: disable=invalid-name
    GObject.type_register(NcmTestModel2)

    model: Ncm.Model = NcmTestModel2()
    model.params_set_default_ftype()

    mset = Ncm.MSet()
    mset.set(model)

    # Test create_params_map without mapping (mapping=None)
    params_map = create_params_map([model_name], mset, None)

    # Verify the function worked correctly
    assert isinstance(params_map, ParamsMap)
    # Should only contain model parameters, not mapping parameters
    assert "test_param2" in params_map
    assert "Omega_c" not in params_map
    assert "h" not in params_map


def test_numcosmo_data_pure_ccl_mode():
    """Test NumCosmoData.do_prepare with PURE_CCL_MODE (non-DEFAULT creation mode)."""
    # This test covers lines 574-577 (the 'else' branch in do_prepare)
    build_parameters = NamedParameters()
    factory = NumCosmoFactory(
        "tests/likelihood/lkdir/lkscript_pure_ccl.py",
        build_parameters,
        None,  # No mapping for PURE_CCL_MODE
        model_list=["test_model_pure_ccl"],
    )

    nc_data = factory.get_data()
    assert isinstance(nc_data, NumCosmoData)

    # Verify that the CCL factory is in PURE_CCL_MODE
    from firecrown.ccl_factory import CCLCreationMode

    assert nc_data.tools.ccl_factory.creation_mode == CCLCreationMode.PURE_CCL_MODE

    # Create a simple model for testing
    model_name = "test_model_pure_ccl"
    model_builder = Ncm.ModelBuilder.new(
        Ncm.Model,
        model_name,
        f"Test model {model_name}",
    )
    # Add required parameters for PURE_CCL_MODE
    params = {
        "Omega_c": 0.26,
        "Omega_b": 0.048,
        "h": 0.7,
        "sigma8": 0.8,
        "n_s": 0.96,
        "Omega_k": 0.0,
        "Neff": 3.046,
        "m_nu": 0.06,
        "w0": -1.0,
        "wa": 0.0,
        "T_CMB": 2.725,
    }

    for param, value in params.items():
        model_builder.add_sparam(
            param,
            param,
            -1.0e10,
            1.0e10,
            1.0e-2,
            0.0,
            value,
            Ncm.ParamType.FIXED,
        )

    TestModelPureCCL = model_builder.create()  # pylint: disable=invalid-name
    GObject.new(TestModelPureCCL)
    NcmTestModelPureCCL = TestModelPureCCL.pytype  # pylint: disable=invalid-name
    GObject.type_register(NcmTestModelPureCCL)

    model: Ncm.Model = NcmTestModelPureCCL()
    model.params_set_default_ftype()

    # Create a basic cosmology model as well
    cosmo = Nc.HICosmoDEXcdm()
    cosmo.add_submodel(Nc.HIPrimPowerLaw.new())
    cosmo.add_submodel(Nc.HIReionCamb.new())

    mset = Ncm.MSet()
    mset.set(cosmo)
    mset.set(model)
    mset.prepare_fparam_map()

    # This should trigger the 'else' branch in do_prepare (lines 574-577)
    # because tools.ccl_factory.creation_mode != CCLCreationMode.DEFAULT
    nc_data.prepare(mset)

    # Verify the preparation worked
    assert nc_data.dof == nc_data.len - mset.fparams_len()


def test_numcosmo_gauss_cov_none_mapping():
    """Test NumCosmoGaussCov._will_calculate_power_spectra with None mapping."""
    # This test covers line 677 where _nc_mapping is None and returns False
    build_parameters = NamedParameters()
    factory = NumCosmoFactory(
        "tests/likelihood/gauss_family/lkscript_const_gaussian.py",
        build_parameters,
        None,  # No mapping - this will trigger _nc_mapping is None
        model_list=["test_model_gauss_none"],
    )

    nc_data = factory.get_data()
    assert isinstance(nc_data, NumCosmoGaussCov)

    # Verify that _nc_mapping is None
    assert nc_data._nc_mapping is None

    # Test _will_calculate_power_spectra method indirectly
    # When _nc_mapping is None, _will_calculate_power_spectra should return False
    # This is tested through the _configure_object method which calls it
    assert nc_data._will_calculate_power_spectra() is False

    # Verify the object was configured correctly
    assert nc_data.len > 0
    assert nc_data.dof > 0


def test_numcosmo_gauss_cov_pure_ccl_mode():
    """Test do_prepare with PURE_CCL_MODE (non-DEFAULT creation mode)."""
    # This test covers lines 887-891 (the 'else' branch in do_prepare)
    build_parameters = NamedParameters({"projection": "harmonic"})
    factory = NumCosmoFactory(
        "tests/likelihood/gauss_family/lkscript_two_point_pure_ccl.py",
        build_parameters,
        None,  # No mapping for PURE_CCL_MODE
        model_list=["test_model_gauss_pure_ccl"],
    )

    nc_data = factory.get_data()
    assert isinstance(nc_data, NumCosmoGaussCov)

    # Verify that the CCL factory is in PURE_CCL_MODE
    from firecrown.ccl_factory import CCLCreationMode

    assert nc_data.tools.ccl_factory.creation_mode == CCLCreationMode.PURE_CCL_MODE

    # Verify that _nc_mapping is None (this triggers assert _nc_mapping is None
    # in line 888)
    assert nc_data._nc_mapping is None

    # Create a simple model for testing
    model_name = "test_model_gauss_pure_ccl"
    model_builder = Ncm.ModelBuilder.new(
        Ncm.Model,
        model_name,
        f"Test model {model_name}",
    )

    # Add required parameters for PURE_CCL_MODE
    params = {
        "Omega_c": 0.26,
        "Omega_b": 0.048,
        "h": 0.7,
        "sigma8": 0.8,
        "n_s": 0.96,
        "Omega_k": 0.0,
        "Neff": 3.046,
        "m_nu": 0.06,
        "w0": -1.0,
        "wa": 0.0,
        "T_CMB": 2.725,
    }

    for param, value in params.items():
        model_builder.add_sparam(
            param,
            param,
            -1.0e10,
            1.0e10,
            1.0e-2,
            0.0,
            value,
            Ncm.ParamType.FIXED,
        )

    TestModelGaussPureCCL = model_builder.create()  # pylint: disable=invalid-name
    GObject.new(TestModelGaussPureCCL)
    NcmTestModelGaussPureCCL = (
        TestModelGaussPureCCL.pytype
    )  # pylint: disable=invalid-name
    GObject.type_register(NcmTestModelGaussPureCCL)

    model: Ncm.Model = NcmTestModelGaussPureCCL()
    model.params_set_default_ftype()

    # Create a basic cosmology model as well
    cosmo = Nc.HICosmoDEXcdm()
    cosmo.add_submodel(Nc.HIPrimPowerLaw.new())
    cosmo.add_submodel(Nc.HIReionCamb.new())

    mset = Ncm.MSet()
    mset.set(cosmo)
    mset.set(model)
    mset.prepare_fparam_map()

    # This should trigger the 'else' branch in do_prepare (lines 887-891)
    # because tools.ccl_factory.creation_mode != CCLCreationMode.DEFAULT
    # and the assert _nc_mapping is None should pass
    nc_data.prepare(mset)

    # Verify the preparation worked
    assert nc_data.dof == nc_data.len - mset.fparams_len()
