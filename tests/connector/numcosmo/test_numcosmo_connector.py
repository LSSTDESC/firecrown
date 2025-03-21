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
