"""Unit tests for the numcosmo connector."""

from numcosmo_py import Ncm, Nc

from firecrown.connector.numcosmo.numcosmo import NumCosmoFactory, MappingNumCosmo

from firecrown.likelihood.likelihood import NamedParameters

Ncm.cfg_init()


def test_numcosmo_connector():
    """Test the NumCosmo connector."""

    map_cosmo = MappingNumCosmo(
        require_nonlinear_pk=True,
        dist=Nc.Distance.new(6.0),
        model_list=["non_existing_model"],
    )
    nc_factory = NumCosmoFactory(
        "tests/likelihood/lkdir/lkscript.py", NamedParameters(), map_cosmo
    )

    assert nc_factory.get_data() is not None
    assert nc_factory.get_mapping() is not None
