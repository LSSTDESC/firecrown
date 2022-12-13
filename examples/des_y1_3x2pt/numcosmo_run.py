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

mb.add_sparam(
    r"ia_\mathrm{bias}", "ia_bias", -5.0, 5.0, 0.1, 0.0, 0.5, Ncm.ParamType.FREE
)
mb.add_sparam(r"\alpha_z", "alphaz", -5.0, 5.0, 0.1, 0.0, 0.0, Ncm.ParamType.FREE)

mb.add_sparam(r"\alpha_g", "alphag", -5.0, 5.0, 0.1, 0.0, -1.0, Ncm.ParamType.FIXED)
mb.add_sparam(r"z_\mathrm{piv}", "z_piv", 0.0, 5.0, 0.1, 0.0, 0.62, Ncm.ParamType.FIXED)

mb.add_sparam(r"lens0_bias", "lens0_bias", 0.8, 3.0, 0.1, 0.0, 1.4, Ncm.ParamType.FREE)
mb.add_sparam(r"lens1_bias", "lens1_bias", 0.8, 3.0, 0.1, 0.0, 1.6, Ncm.ParamType.FREE)
mb.add_sparam(r"lens2_bias", "lens2_bias", 0.8, 3.0, 0.1, 0.0, 1.6, Ncm.ParamType.FREE)
mb.add_sparam(r"lens3_bias", "lens3_bias", 0.8, 3.0, 0.1, 0.0, 1.9, Ncm.ParamType.FREE)
mb.add_sparam(r"lens4_bias", "lens4_bias", 0.8, 3.0, 0.1, 0.0, 2.0, Ncm.ParamType.FREE)

mb.add_sparam(
    r"src0_delta_z",
    "src0_delta_z",
    -0.16,
    +0.16,
    0.001,
    0.0,
    -0.001,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"src1_delta_z",
    "src1_delta_z",
    -0.13,
    +0.13,
    0.001,
    0.0,
    -0.019,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"src2_delta_z",
    "src2_delta_z",
    -0.11,
    +0.11,
    0.001,
    0.0,
    +0.009,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"src3_delta_z",
    "src3_delta_z",
    -0.22,
    +0.22,
    0.001,
    0.0,
    -0.018,
    Ncm.ParamType.FREE,
)

mb.add_sparam(
    r"lens0_delta_z",
    "lens0_delta_z",
    -0.1,
    +0.1,
    0.001,
    0.0,
    +0.001,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"lens1_delta_z",
    "lens1_delta_z",
    -0.1,
    +0.1,
    0.001,
    0.0,
    +0.002,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"lens2_delta_z",
    "lens2_delta_z",
    -0.1,
    +0.1,
    0.001,
    0.0,
    +0.001,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"lens3_delta_z",
    "lens3_delta_z",
    -0.1,
    +0.1,
    0.001,
    0.0,
    +0.003,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"lens4_delta_z",
    "lens4_delta_z",
    -0.1,
    +0.1,
    0.001,
    0.0,
    +0.000,
    Ncm.ParamType.FREE,
)

mb.add_sparam(
    r"src0_mult_bias",
    "src0_mult_bias",
    -0.23,
    +0.23,
    0.001,
    0.0,
    +0.000,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"src1_mult_bias",
    "src1_mult_bias",
    -0.23,
    +0.23,
    0.001,
    0.0,
    +0.000,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"src2_mult_bias",
    "src2_mult_bias",
    -0.23,
    +0.23,
    0.001,
    0.0,
    +0.000,
    Ncm.ParamType.FREE,
)
mb.add_sparam(
    r"src3_mult_bias",
    "src3_mult_bias",
    -0.23,
    +0.23,
    0.001,
    0.0,
    +0.000,
    Ncm.ParamType.FREE,
)

GNcFirecrown = mb.create()
GObject.new(GNcFirecrown)
NcFirecrown = GNcFirecrown.pytype
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

ps_ml = Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new())
ps_mnl = Nc.PowspecMNLHaloFit.new(ps_ml, 3.0, 1.0e-5)
dist = Nc.Distance.new(6.0)
dist.comoving_distance_spline.set_reltol(1.0e-5)

map_cosmo = MappingNumCosmo(
    require_nonlinear_pk=True, p_ml=ps_ml, p_mnl=ps_mnl, dist=dist
)

nc_factory = NumCosmoFactory("des_y1_3x2pt.py", {}, ["NcFirecrown"], map_cosmo)

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
lh.priors_add_gauss_param_name(mset, "NcFirecrown:src1_delta_z", -0.019, 0.013)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:src2_delta_z", +0.009, 0.011)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:src3_delta_z", -0.018, 0.022)

lh.priors_add_gauss_param_name(mset, "NcFirecrown:lens0_delta_z", +0.001, 0.008)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:lens1_delta_z", +0.002, 0.007)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:lens2_delta_z", +0.001, 0.007)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:lens3_delta_z", +0.003, 0.010)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:lens4_delta_z", +0.000, 0.010)

lh.priors_add_gauss_param_name(mset, "NcFirecrown:src0_mult_bias", +0.012, 0.023)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:src0_mult_bias", +0.012, 0.023)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:src0_mult_bias", +0.012, 0.023)
lh.priors_add_gauss_param_name(mset, "NcFirecrown:src0_mult_bias", +0.012, 0.023)

fit = Ncm.Fit.new(
    Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD
)

mset.pretty_log()
fit.run_restart(Ncm.FitRunMsgs.FULL, 1.0e-3, 0.0, None, None)
