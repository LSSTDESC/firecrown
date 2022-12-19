#!/usr/bin/env python

from firecrown.connector.numcosmo.numcosmo import MappingNumCosmo, NumCosmoFactory

import math
import gi

gi.require_version("NumCosmo", "1.0")
gi.require_version("NumCosmoMath", "1.0")

from gi.repository import GObject  # noqa: E402
from gi.repository import NumCosmo as Nc  # noqa: E402
from gi.repository import NumCosmoMath as Ncm  # noqa: E402

import matplotlib.pyplot as plt
from scipy.stats import chi2

Ncm.cfg_init()

mb = Ncm.ModelBuilder.new(Ncm.Model, "NcFirecrown", "Firecrown model interface")
ser = Ncm.Serialize.new(Ncm.SerializeOpt.NONE)
sparams = Ncm.ObjArray.load("numcosmo_firecrown_model.oa", ser)
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

_, H0_i = cosmo.param_index_from_name("H0")
cosmo.param_set_ftype(H0_i, Ncm.ParamType.FREE)

_, Omegac_i = cosmo.param_index_from_name("Omegac")
cosmo.param_set_ftype(Omegac_i, Ncm.ParamType.FREE)

prim = Nc.HIPrimPowerLaw.new()
prim.param_set_by_name("ln10e10ASA", math.log(1.0e10 * 2.0e-09))
prim.param_set_by_name("n_SA", 0.971)

reion = Nc.HIReionCamb.new()
reion.set_z_from_tau(cosmo, 0.0561)

cosmo.add_submodel(prim)
cosmo.add_submodel(reion)

dist = Nc.Distance.new(6.0)
dist.comoving_distance_spline.set_reltol(1.0e-5)

map_cosmo = MappingNumCosmo(require_nonlinear_pk=False, dist=dist)

nc_factory = NumCosmoFactory("sn_srd.py", {}, ["NcFirecrown"], map_cosmo)

fc = NcFirecrown()
# fc.params_set_default_ftype()

mset = Ncm.MSet()
mset.set(cosmo)
mset.set(fc)

fc_data = nc_factory.get_data()

dset = Ncm.Dataset()
dset.append_data(fc_data)

lh = Ncm.Likelihood(dataset=dset)

fit = Ncm.Fit.new(
    Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD
)

mset.pretty_log()
fit.run_restart(Ncm.FitRunMsgs.SIMPLE, 1.0e-3, 0.0, None, None)
fit.fisher()
fit.log_covar()

p1 = Ncm.MSetPIndex.new(cosmo.id(), H0_i)
p2 = Ncm.MSetPIndex.new(cosmo.id(), Omegac_i)

lhr2d = Ncm.LHRatio2d.new(fit, p1, p2, 1.0e-3)

plt.figure(figsize=(8, 4))
plt.title("Confidence regions")

for clevel in [chi2.cdf(l**2, df=1) for l in [1, 2, 3]]:
    fisher_rg = lhr2d.fisher_border(clevel, 300.0, Ncm.FitRunMsgs.SIMPLE)
    plt.plot(
        fisher_rg.p1.dup_array(),
        fisher_rg.p2.dup_array(),
        label=f"Fisher Matrix -- {fisher_rg.clevel*100:.2f}%",
    )

plt.xlabel(f"${cosmo.param_symbol(H0_i)}$")
plt.ylabel(f"${cosmo.param_symbol(Omegac_i)}$")

plt.legend(loc="best")

plt.savefig("srd_sn_fisher_example.pdf")
