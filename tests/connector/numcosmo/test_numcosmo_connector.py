"""Unit tests for the numcosmo connector."""

import pytest
from numcosmo_py import Ncm, Nc

from firecrown.connector.numcosmo.numcosmo import (
    NumCosmoFactory,
    MappingNumCosmo,
    NumCosmoData,
    NumCosmoGaussCov,
)

from firecrown.likelihood.likelihood import NamedParameters, Likelihood

Ncm.cfg_init()


@pytest.fixture(name="factory_plain")
def fixture_factory_plain():
    """Create a NumCosmoFactory instance."""
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
    )
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
    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
    )
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
