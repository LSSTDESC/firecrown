#!/usr/bin/env python

"""Defines a function to generate a SACC file for cluster number counts."""

# # Cluster count-only SACC file creation
#
# This notebook examplifies the creation of a SACC file for cluster count, using
# NumCosmo facilities to simulate cluster data.

import math
import numpy as np

from numcosmo_py import Nc
from numcosmo_py import Ncm

from astropy.table import Table

from astropy.io import fits
from scipy import stats
from typing import Any

import os

os.environ["CLMM_MODELING_BACKEND"] = (
    "nc"  # Need to use NumCosmo as CLMM's backend as well.
)
import clmm  # noqa: E402
from clmm import Cosmology  # noqa: E402

from gen_sacc_aux import convert_binned_profile_to_sacc  # noqa: E402


def generate_sacc_file() -> Any:
    """Generate a SACC file for cluster number counts."""
    H0 = 71.0
    Ob0 = 0.0448
    Odm0 = 0.22
    n_s = 0.963
    sigma8 = 0.8

    cosmo = Nc.HICosmoDECpl()
    reion = Nc.HIReionCamb.new()
    prim = Nc.HIPrimPowerLaw.new()

    cosmo.add_submodel(reion)
    cosmo.add_submodel(prim)

    dist = Nc.Distance.new(2.0)
    tf = Nc.TransferFuncEH.new()

    psml = Nc.PowspecMLTransfer.new(tf)

    # psml = Nc.PowspecMLCBE.new ()
    psml.require_kmin(1.0e-6)
    psml.require_kmax(1.0e3)

    psf = Ncm.PowspecFilter.new(psml, Ncm.PowspecFilterType.TOPHAT)
    psf.set_best_lnr0()

    cosmo.props.H0 = H0
    cosmo.props.Omegab = Ob0
    cosmo.props.Omegac = Odm0

    cosmo.omega_x2omega_k()
    cosmo.param_set_by_name("Omegak", 0.0)

    prim.props.n_SA = n_s

    old_amplitude = math.exp(prim.props.ln10e10ASA)
    prim.props.ln10e10ASA = math.log((sigma8 / cosmo.sigma8(psf)) ** 2 * old_amplitude)

    # CosmoSim_proxy model
    # M_0, z_0

    area = 439.78986
    lnRl = 0.0
    lnRu = 5.0
    zl = 0.2
    zu = 0.65

    # NumCosmo proxy model based on arxiv 1904.07524v2
    cluster_z = Nc.ClusterRedshiftNodist(z_max=zu, z_min=zl)
    cluster_m = Nc.ClusterMassAscaso(
        M0=3.0e14 / 0.71, z0=0.6, lnRichness_min=lnRl, lnRichness_max=lnRu
    )
    cluster_m.param_set_by_name("mup0", 3.19)
    cluster_m.param_set_by_name("mup1", 2 / np.log(10))
    cluster_m.param_set_by_name("mup2", -0.7 / np.log(10))
    cluster_m.param_set_by_name("sigmap0", 0.33)
    cluster_m.param_set_by_name("sigmap1", -0.08 / np.log(10))
    cluster_m.param_set_by_name("sigmap2", 0 / np.log(10))

    # Numcosmo Mass Function
    # First we need to define the multiplicity function here we will use the tinker
    mulf = Nc.MultiplicityFuncTinker.new()
    mulf.set_linear_interp(True)  # This reproduces the linear interpolation done in CCL
    mulf.set_mdef(Nc.MultiplicityFuncMassDef.MEAN)
    mulf.set_Delta(200)

    # Second we need to construct a filtered power spectrum
    hmf = Nc.HaloMassFunction.new(dist, psf, mulf)
    hmf.set_area_sd(area)

    # Cluster Abundance Obj
    ca = Nc.ClusterAbundance.new(hmf, None)

    # Number Counts object
    ncount = Nc.DataClusterNCount.new(
        ca, "NcClusterRedshiftNodist", "NcClusterMassAscaso"
    )
    ca.prepare(cosmo, cluster_z, cluster_m)
    mset = Ncm.MSet.new_array([cosmo, cluster_z, cluster_m])

    rng = Ncm.RNG.seeded_new(None, 32)
    ncount.init_from_sampling(mset, area * ((np.pi / 180) ** 2), rng)

    ncount.catalog_save("ncount_rich.fits", True)
    ncdata_fits = fits.open("ncount_rich.fits")
    ncdata_data = ncdata_fits[1].data  # pylint: disable-msg=no-member
    ncdata_Table = Table(ncdata_data)

    # ## Saving in SACC format

    data_table = ncdata_Table[ncdata_Table["LNM_OBS"] > 2]
    cluster_z = data_table["Z_OBS"]
    cluster_lnm = data_table["LNM_OBS"]
    cluster_richness = cluster_lnm / np.log(10.0)
    cluster_logM = data_table["LNM_TRUE"] / np.log(10.0)

    # ## Count halos in the $N_{\rm richness} \times N_z$ richness-redshift plane

    N_richness = 5  # number of richness bins
    N_z = 4  # number of redshift bins

    cluster_counts, z_edges, richness_edges, _ = stats.binned_statistic_2d(
        cluster_z, cluster_richness, cluster_logM, "count", bins=[N_z, N_richness]
    )

    # Use CLMM to create a mock DeltaSigma profile to add to the SACC file later
    cosmo_clmm = Cosmology()
    cosmo_clmm._init_from_cosmo(cosmo)
    moo = clmm.Modeling(massdef="mean", delta_mdef=200, halo_profile_model="nfw")
    moo.set_cosmo(cosmo_clmm)
    # assuming the same concentration for all masses. Not realistic, but avoid having
    # to call a mass-concentration relation.
    moo.set_concentration(4)

    # Make data
    radius_edges = clmm.make_bins(
        0.3, 6.0, nbins=6, method="evenlog10width"
    )  # 6 radial bins log-spaced between 0.3 and 6 Mpc
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])

    cluster_DeltaSigma = []
    for redshift, log_mass in zip(cluster_z, cluster_logM):
        mass = 10**log_mass
        moo.set_mass(mass)
        cluster_DeltaSigma.append(
            moo.eval_excess_surface_density(radius_centers, redshift)
        )
    cluster_DeltaSigma = np.log10(np.array(cluster_DeltaSigma))

    richness_inds = np.digitize(cluster_richness, richness_edges) - 1
    z_inds = np.digitize(cluster_z, z_edges) - 1
    mean_DeltaSigma = np.array(
        [
            [
                np.mean(
                    cluster_DeltaSigma[(richness_inds == i) * (z_inds == j)], axis=0
                )
                for i in range(N_richness)
            ]
            for j in range(N_z)
        ]
    )
    std_DeltaSigma = np.array(
        [
            [
                np.std(cluster_DeltaSigma[(richness_inds == i) * (z_inds == j)], axis=0)
                for i in range(N_richness)
            ]
            for j in range(N_z)
        ]
    )

    var_mean_DeltaSigma = std_DeltaSigma**2 / cluster_counts[..., None]

    # ** Correlation matrix - the "large blocks" correspond to the $N_z$ redshift bins.
    # In each redshift bin are the $N_{\rm richness}$ richness bins.**

    covariance = np.diag(
        np.concatenate((cluster_counts.flatten(), var_mean_DeltaSigma.flatten()))
    )
    s_count = convert_binned_profile_to_sacc(
        cluster_counts,
        mean_DeltaSigma,
        covariance,
        z_edges,
        richness_edges,
        radius_edges,
        radius_centers,
        area,
    )
    s_count.save_fits("cluster_redshift_richness_sacc_data.fits", overwrite=True)


if __name__ == "__main__":
    Ncm.cfg_init()
    generate_sacc_file()
