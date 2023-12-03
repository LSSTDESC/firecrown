#!/usr/bin/env python
"""Example of running the DES Y1 3x2pt likelihood using NumCosmo."""

import math
import os
from typing import Tuple
import argparse
from pathlib import Path


import yaml

from numcosmo_py import Nc, Ncm
from numcosmo_py.sampling.esmcmc import create_esmcmc

from firecrown.connector.numcosmo.numcosmo import MappingNumCosmo, NumCosmoFactory
from firecrown.connector.numcosmo.model import define_numcosmo_model
from firecrown.likelihood.likelihood import NamedParameters

# Any NumCosmo model should be loaded before NumCosmo cfg_init
# is called. Otherwise, the NumCosmo model will not be registered
# in the model set. This is required for the NumCosmo MPI support
# to work properly.


module_path = Path(os.path.dirname(__file__))
with open(
    module_path / r"numcosmo_firecrown_model.yml", "r", encoding="utf-8"
) as modelfile:
    ncmodel = yaml.load(modelfile, Loader=yaml.Loader)

NcFirecrown = define_numcosmo_model(ncmodel)


def setup_numcosmo_cosmology() -> Tuple[Nc.HICosmo, MappingNumCosmo]:
    """Setup NumCosmo cosmology.

    Creates a NumCosmo cosmology object.
    """

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
    )

    return cosmo, map_cosmo


def setup_firecrown(map_cosmo: MappingNumCosmo) -> Tuple[Ncm.Model, NumCosmoFactory]:
    """Setup Firecrown object."""

    nc_factory = NumCosmoFactory(
        str(module_path / "des_y1_3x2pt.py"),
        NamedParameters(),
        map_cosmo,
        [ncmodel.name],
    )

    fc = NcFirecrown()

    return fc, nc_factory


def setup_likelihood(
    cosmo: Nc.HICosmo,
    fc: Ncm.Model,
    nc_factory: NumCosmoFactory,
) -> Tuple[Ncm.Likelihood, Ncm.MSet]:
    """Setup the likelihood and model set objects."""

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

    return lh, mset


def setup_fit(likelihood: Ncm.Likelihood, mset: Ncm.MSet) -> Ncm.Fit:
    """Setup the fit object."""

    fit = Ncm.Fit.factory(
        Ncm.FitType.NLOPT,
        "ln-neldermead",
        likelihood,
        mset,
        Ncm.FitGradType.NUMDIFF_FORWARD,
    )

    return fit


def setup_fit_all() -> Ncm.Fit:
    """Setup all objects necessary to instantiate the fit object."""

    cosmo, map_cosmo = setup_numcosmo_cosmology()
    fc, nc_factory = setup_firecrown(map_cosmo)
    likelihood, mset = setup_likelihood(cosmo, fc, nc_factory)
    fit = setup_fit(likelihood, mset)

    return fit


def run_test() -> None:
    """Run the fit."""

    fit = setup_fit_all()
    mset = fit.peek_mset()
    mset.param_set_all_ftype(Ncm.ParamType.FIXED)
    mset.pretty_log()

    fit.run_restart(Ncm.FitRunMsgs.FULL, 1.0e-3, 0.0, None, None)


def run_compute_best_fit() -> None:
    """Run the fit."""

    fit = setup_fit_all()
    mset = fit.peek_mset()

    # Sets the default ftype for all model parameters
    for i in range(mset.nmodels()):
        model = mset.peek_array_pos(i)
        model.params_set_default_ftype()
    mset.prepare_fparam_map()
    mset.pretty_log()

    fit.run_restart(Ncm.FitRunMsgs.FULL, 1.0e-3, 0.0, None, None)


def run_apes_sampler(ssize: int) -> None:
    """Run the fit."""

    fit = setup_fit_all()
    mset = fit.peek_mset()

    # Sets the default ftype for all model parameters
    for i in range(mset.nmodels()):
        model = mset.peek_array_pos(i)
        model.params_set_default_ftype()
    mset.prepare_fparam_map()

    nwalkers = mset.fparam_len * 100
    esmcmc = create_esmcmc(
        fit.likelihood, mset, "des_y1_3x2pt_apes", nwalkers=nwalkers, nthreads=1
    )

    esmcmc.start_run()
    esmcmc.run(ssize // nwalkers)
    esmcmc.end_run()


if __name__ == "__main__":
    Ncm.cfg_init()

    parser = argparse.ArgumentParser(description="Run DES Y1 3x2pt likelihood.")

    parser.add_argument(
        "--run-mode",
        choices=["test_likelihood", "compute_best_fit", "run_apes_sampler"],
        default="test_likelihood",
        help="Run mode.",
    )

    parser.add_argument(
        "--apes-sampler-ssize",
        type=int,
        default=10,
        help="Number of samples to draw from the APES sampler.",
    )

    args = parser.parse_args()

    if args.run_mode == "test_likelihood":
        run_test()
    elif args.run_mode == "compute_best_fit":
        run_compute_best_fit()
    elif args.run_mode == "run_apes_sampler":
        run_apes_sampler(args.apes_sampler_ssize)
    else:
        raise ValueError(f"Unknown run mode: {args.run_mode}")
