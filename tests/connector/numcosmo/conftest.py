"""Collection of fixtures for the numcosmo connector tests."""

from typing import cast
import math
import pytest

from numcosmo_py import Nc, Ncm, GObject


@pytest.fixture(name="numcosmo_cosmo_xcdm")
def fixture_numcosmo_cosmo_xcdm():
    """Create a NumCosmo cosmology instance of XCDM."""
    cosmo = Nc.HICosmoDEXcdm(massnu_length=1)
    cosmo.omega_x2omega_k()
    cosmo.param_set_by_name("H0", 68.2)
    cosmo.param_set_by_name("Omegak", 0.0)
    cosmo.param_set_by_name("Omegab", 0.022558514 / 0.682**2)
    cosmo.param_set_by_name("Omegac", 0.118374058 / 0.682**2)
    cosmo.param_set_by_name("massnu_0", 0.06)
    cosmo.param_set_by_name("ENnu", 2.0328)
    cosmo.param_set_by_name("Yp", 0.2454)
    cosmo.param_set_by_name("w", -1.0)

    prim = Nc.HIPrimPowerLaw.new()
    prim.param_set_by_name(  # pylint: disable=no-member
        "ln10e10ASA", math.log(1.0e10 * 2.0e-09)
    )
    prim.param_set_by_name("n_SA", 0.971)  # pylint: disable=no-member

    reion = Nc.HIReionCamb.new()
    reion.set_z_from_tau(cosmo, 0.0561)  # pylint: disable=no-member

    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)

    p_ml = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
    p_mnl = Nc.PowspecMNLHaloFit.new(p_ml, 3.0, 1.0e-5)
    dist = Nc.Distance.new(6.0)
    dist.comoving_distance_spline.set_reltol(1.0e-5)  # pylint: disable=no-member

    return {"cosmo": cosmo, "p_ml": p_ml, "p_mnl": p_mnl, "dist": dist}


@pytest.fixture(name="numcosmo_cosmo_xcdm_no_nu")
def fixture_numcosmo_cosmo_xcdm_no_nu():
    """Create a NumCosmo cosmology instance of XCDM without neutrinos."""
    cosmo = Nc.HICosmoDEXcdm()
    cosmo.omega_x2omega_k()
    cosmo.param_set_by_name("H0", 68.2)
    cosmo.param_set_by_name("Omegak", 0.0)
    cosmo.param_set_by_name("Omegab", 0.022558514 / 0.682**2)
    cosmo.param_set_by_name("Omegac", 0.118374058 / 0.682**2)
    cosmo.param_set_by_name("ENnu", 3.046)
    cosmo.param_set_by_name("Yp", 0.2454)
    cosmo.param_set_by_name("w", -1.0)

    prim = Nc.HIPrimPowerLaw.new()
    prim.param_set_by_name(  # pylint: disable=no-member
        "ln10e10ASA", math.log(1.0e10 * 2.0e-09)
    )
    prim.param_set_by_name("n_SA", 0.971)  # pylint: disable=no-member

    reion = Nc.HIReionCamb.new()
    reion.set_z_from_tau(cosmo, 0.0561)  # pylint: disable=no-member

    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)

    p_ml = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
    p_mnl = Nc.PowspecMNLHaloFit.new(p_ml, 3.0, 1.0e-5)
    dist = Nc.Distance.new(6.0)
    dist.comoving_distance_spline.set_reltol(1.0e-5)  # pylint: disable=no-member

    return {"cosmo": cosmo, "p_ml": p_ml, "p_mnl": p_mnl, "dist": dist}


@pytest.fixture(name="numcosmo_cosmo_cpl")
def fixture_numcosmo_cosmo_cpl():
    """Create a NumCosmo cosmology instance using CPL parametrization for the dark
    energy equation of state."""

    cosmo = Nc.HICosmoDECpl()
    cosmo.omega_x2omega_k()
    cosmo.param_set_by_name("H0", 68.2)
    cosmo.param_set_by_name("Omegak", 0.0)
    cosmo.param_set_by_name("Omegab", 0.022558514 / 0.682**2)
    cosmo.param_set_by_name("Omegac", 0.118374058 / 0.682**2)
    cosmo.param_set_by_name("ENnu", 3.046)
    cosmo.param_set_by_name("Yp", 0.2454)
    cosmo.param_set_by_name("w0", -1.0)
    cosmo.param_set_by_name("w1", 0.1)

    prim = Nc.HIPrimPowerLaw.new()
    prim.param_set_by_name(  # pylint: disable=no-member
        "ln10e10ASA", math.log(1.0e10 * 2.0e-09)
    )
    prim.param_set_by_name("n_SA", 0.971)  # pylint: disable=no-member

    reion = Nc.HIReionCamb.new()
    reion.set_z_from_tau(cosmo, 0.0561)  # pylint: disable=no-member

    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)

    p_ml = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
    p_mnl = Nc.PowspecMNLHaloFit.new(p_ml, 3.0, 1.0e-5)
    dist = Nc.Distance.new(6.0)
    dist.comoving_distance_spline.set_reltol(1.0e-5)  # pylint: disable=no-member

    return {"cosmo": cosmo, "p_ml": p_ml, "p_mnl": p_mnl, "dist": dist}


@pytest.fixture(name="nc_model_trivial", scope="session")
def fixture_nc_model_trivial():
    """Create a NumCosmoModel instance."""
    my_model_trivial_yaml = r"""
NcmModelBuilder:
    parent-type-string: 'NcmModel'
    name: 'NcFirecrownTrivial'
    description: 'Trivial model description'
    sparams:
    - NcmSParam:
        name: 'mean'
        symbol: '\mu'
        lower-bound: -5.0
        upper-bound: 5.0
        scale: 1.0
        absolute-tolerance: 0.0
        default-value: 0.1
        fit-type: 0
"""
    ser = Ncm.Serialize.new(Ncm.SerializeOpt.NONE)
    mb_model: Ncm.ModelBuilder = cast(
        Ncm.ModelBuilder,
        ser.from_yaml(my_model_trivial_yaml),  # pylint: disable=no-member
    )
    assert isinstance(mb_model, Ncm.ModelBuilder)
    model_type = mb_model.create()
    GObject.new(model_type)

    return model_type.pytype
