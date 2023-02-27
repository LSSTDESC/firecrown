#!/usr/bin/env python

from firecrown.connector.numcosmo.numcosmo import MappingNumCosmo, NumCosmoFactory

import math
import gi

gi.require_version("NumCosmo", "1.0")
gi.require_version("NumCosmoMath", "1.0")

from gi.repository import GObject  # noqa: E402
from gi.repository import NumCosmo as Nc  # noqa: E402
from gi.repository import NumCosmoMath as Ncm  # noqa: E402

Ncm.cfg_init()

mb = Ncm.ModelBuilder.new(Ncm.Model, "NcFirecrown", "Firecrown model interface")
ser = Ncm.Serialize.new(Ncm.SerializeOpt.NONE)
sparams = Ncm.ObjArray.load("numcosmo_firecrown_model_PT.oa", ser)
for i in range(sparams.len()):
    mb.add_sparam_obj(sparams.get(i))

NcTypeFirecrown = mb.create()
GObject.new(NcTypeFirecrown)
NcFirecrown = NcTypeFirecrown.pytype
GObject.type_register(NcFirecrown)

cosmo = Nc.HICosmo.new_from_name(Nc.HICosmo, "NcHICosmoDEXcdm{'massnu-length':<1>}")
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
    require_nonlinear_pk=True, p_ml=p_ml, p_mnl=p_mnl, dist=dist
)

nc_factory = NumCosmoFactory("des_y1_3x2pt_PT.py", {}, ["NcFirecrown"], map_cosmo)

fc = NcFirecrown()
# fc.params_set_default_ftype()

mset = Ncm.MSet()
mset.set(cosmo)
mset.set(fc)

fc_data = nc_factory.get_data()

dset = Ncm.Dataset()
dset.append_data(fc_data)

lh = Ncm.Likelihood(dataset=dset)

lh.priors_add_gauss_param_name(mset, "NcFirecrown:src0_delta_z", -0.001, 0.016)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:lens0_delta_z", +0.001, 0.008)

fit = Ncm.Fit.new(
    Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD
)

mset.pretty_log()
fit.run_restart(Ncm.FitRunMsgs.FULL, 1.0e-3, 0.0, None, None)
