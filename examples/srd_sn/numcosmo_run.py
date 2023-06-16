#!/usr/bin/env python
"""
Numcosmo example using the :python:`sn_srd` likelihood.
"""

import math
import yaml

import matplotlib.pyplot as plt
from scipy.stats import chi2
from numcosmo_py import Nc, Ncm

from firecrown.connector.numcosmo.numcosmo import (
    MappingNumCosmo,
    NumCosmoFactory,
)
from firecrown.connector.numcosmo.model import define_numcosmo_model
from firecrown.likelihood.likelihood import NamedParameters

Ncm.cfg_init()

with open(r"numcosmo_firecrown_model_snia.yml", "r", encoding="utf8") as modelfile:
    ncmodel = yaml.load(modelfile, Loader=yaml.Loader)

NcFirecrownSNIa = define_numcosmo_model(ncmodel)

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

map_cosmo = MappingNumCosmo(
    require_nonlinear_pk=False, dist=dist, model_list=["NcFirecrownSNIa"]
)

nc_factory = NumCosmoFactory("sn_srd.py", NamedParameters(), map_cosmo)

fc = NcFirecrownSNIa()
# fc.params_set_default_ftype()

mset = Ncm.MSet()
mset.set(cosmo)
mset.set(fc)

fc_data = nc_factory.get_data()

dset = Ncm.Dataset()
dset.append_data(fc_data)

lh = Ncm.Likelihood(dataset=dset)

fit = Ncm.Fit.new(
    Ncm.FitType.NLOPT,
    "ln-neldermead",
    lh,
    mset,
    Ncm.FitGradType.NUMDIFF_FORWARD,
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

for clevel in [chi2.cdf(chi**2, df=1) for chi in [1, 2, 3]]:
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
