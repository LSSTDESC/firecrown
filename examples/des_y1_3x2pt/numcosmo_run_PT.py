#!/usr/bin/env python

"""Example of running the DES Y1 3x2pt likelihood using NumCosmo."""

import math

import yaml

from numcosmo_py import Nc, Ncm

from firecrown.connector.numcosmo.numcosmo import MappingNumCosmo, NumCosmoFactory
from firecrown.connector.numcosmo.model import define_numcosmo_model
from firecrown.likelihood.likelihood import NamedParameters


Ncm.cfg_init()

with open(r"numcosmo_firecrown_model_PT.yml", "r", encoding="utf-8") as modelfile:
    ncmodel = yaml.load(modelfile, Loader=yaml.Loader)

NcFirecrownPT = define_numcosmo_model(ncmodel)

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
prim.param_set_by_name("ln10e10ASA", math.log(1.0e10 * 2.0e-09))
prim.param_set_by_name("n_SA", 0.971)

reion = Nc.HIReionCamb.new()
reion.set_z_from_tau(cosmo, 0.0561)

cosmo.add_submodel(prim)
cosmo.add_submodel(reion)

p_ml = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
p_mnl = Nc.PowspecMNLHaloFit.new(p_ml, 3.0, 1.0e-5)
dist = Nc.Distance.new(6.0)
dist.comoving_distance_spline.set_reltol(1.0e-5)

map_cosmo = MappingNumCosmo(
    require_nonlinear_pk=True,
    p_ml=p_ml,
    p_mnl=p_mnl,
    dist=dist,
    model_list=["NcFirecrownPT"],
)

nc_factory = NumCosmoFactory("des_y1_3x2pt_PT.py", NamedParameters(), map_cosmo)

fc = NcFirecrownPT()
# fc.params_set_default_ftype()

mset = Ncm.MSet()
mset.set(cosmo)
mset.set(fc)

fc_data = nc_factory.get_data()

dset = Ncm.Dataset()
dset.append_data(fc_data)

lh = Ncm.Likelihood(dataset=dset)

lh.priors_add_gauss_param_name(mset, "NcFirecrownPT:src0_delta_z", -0.001, 0.016)
lh.priors_add_gauss_param_name(mset, "NcFirecrownPT:lens0_delta_z", +0.001, 0.008)

fit = Ncm.Fit.new(
    Ncm.FitType.NLOPT,
    "ln-neldermead",
    lh,
    mset,
    Ncm.FitGradType.NUMDIFF_FORWARD,
)

mset.pretty_log()
fit.run_restart(Ncm.FitRunMsgs.FULL, 1.0e-3, 0.0, None, None)
