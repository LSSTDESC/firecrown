#!/usr/bin/env python

"""Function to generate SACC file for cluster number counts and cluster deltasigma."""

# # Cluster count-only SACC file creation
#
# This file examplifies the creation of a SACC file for cluster count, using
# NumCosmo facilities to simulate cluster data.

import math
import itertools

import numpy as np

from numcosmo_py import Nc
from numcosmo_py import Ncm

from astropy.table import Table

from astropy.io import fits
from scipy import stats
from typing import Any
import sacc
import pyccl as ccl
import os

os.environ["CLMM_MODELING_BACKEND"] = (
    "ccl"  # Need to use NumCosmo as CLMM's backend as well.
)
import clmm  # noqa: E402
from clmm import Cosmology  # noqa: E402


def generate_sacc_file(
    H0: float = 71.0,
    Ob0: float = 0.0448,
    Odm0: float = 0.22,
    n_s: float = 0.963,
    sigma8: float = 0.8,
    area: float = 439.78986,
    M0: float = 3.0e14 / 0.71,
    z0: float = 0.6,
    lnRl: float = 0.0,
    lnRu: float = 5.0,
    zl: float = 0.2,
    zu: float = 0.65,
) -> None:
    """Generate a SACC file for cluster number counts and cluster deltasigma.

    Set up the cosmological model with specified parameters.

    :param H0: Hubble constant in km/s/Mpc.
    :param Ob0: Baryon density parameter.
    :param Odm0: Dark matter density parameter.
    :param n_s: Scalar spectral index.
    :param sigma8: The amplitude of matter fluctuations on scales of 8 Mpc/h.
    :param area: Survey area in square degrees.
    :param zl: Minimum redshift for clusters.
    :param zu: Maximum redshift for clusters.
    :param M0: Characteristic mass at z=0 in units of M_sun/h.
    :param z0: Characteristic redshift for mass evolution.
    :param lnRl: Minimum natural log of cluster richness.
    :param lnRu: Maximum natural log of cluster richness.
    """
    cosmo = Nc.HICosmoDECpl()
    cosmo_ccl = ccl.Cosmology(
        Omega_b=Ob0, Omega_c=Odm0, sigma8=sigma8, w0=-1, wa=0, h=H0 / 100.0, n_s=n_s
    )
    reion = Nc.HIReionCamb.new()
    prim = Nc.HIPrimPowerLaw.new()

    cosmo.add_submodel(reion)
    cosmo.add_submodel(prim)

    dist = Nc.Distance.new(2.0)
    tf = Nc.TransferFuncEH.new()

    psml = Nc.PowspecMLTransfer.new(tf)

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

    # NumCosmo proxy model based on arxiv 1904.07524v2
    cluster_z = Nc.ClusterRedshiftNodist(z_max=zu, z_min=zl)
    cluster_m = Nc.ClusterMassAscaso(
        M0=M0, z0=z0, lnRichness_min=lnRl, lnRichness_max=lnRu
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
    mulf.set_mdef(Nc.MultiplicityFuncMassDef.CRITICAL)
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

    rng = Ncm.RNG.seeded_new(None, 2)
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
    mean_logM = stats.binned_statistic_2d(
        cluster_z,
        cluster_richness,
        cluster_logM,
        "mean",
        bins=[z_edges, richness_edges],
    ).statistic
    std_logM = stats.binned_statistic_2d(
        cluster_z, cluster_richness, cluster_logM, "std", bins=[z_edges, richness_edges]
    ).statistic

    var_mean_logM = std_logM**2 / cluster_counts
    # Use CLMM to create a mock DeltaSigma profile to add to the SACC file later
    cosmo_clmm = Cosmology()
    cosmo_clmm._init_from_cosmo(cosmo_ccl)
    moo = clmm.Modeling(massdef="critical", delta_mdef=200, halo_profile_model="nfw")
    moo.set_cosmo(cosmo_clmm)
    # assuming the same concentration for all masses. Not realistic,
    # but avoid having to call a mass-concentration relation.
    moo.set_concentration(4)

    radius_edges = clmm.make_bins(
        0.3, 6.0, nbins=6, method="evenlog10width"
    )  # 6 radial bins log-spaced between 0.3 and 6 Mpc

    radius_centers = []
    for i, radius_bin in enumerate(zip(radius_edges[:-1], radius_edges[1:])):
        radius_lower, radius_upper = radius_bin
        j = i + 2
        radius_center = np.mean(radius_edges[i:j])
        radius_centers.append(radius_center)

    cluster_DeltaSigma_list = []
    for redshift, log_mass in zip(cluster_z, cluster_logM):
        mass = 10**log_mass
        moo.set_mass(mass)
        cluster_DeltaSigma_list.append(
            moo.eval_excess_surface_density(radius_centers, redshift)
        )
    cluster_DeltaSigma = np.array(cluster_DeltaSigma_list)
    mean_DeltaSigma = np.zeros((N_z, N_richness, len(radius_edges) - 1))
    std_DeltaSigma = np.zeros((N_z, N_richness, len(radius_edges) - 1))
    for i, radius_bin in enumerate(radius_edges[:-1]):
        cluster_DeltaSigma_at_radius = cluster_DeltaSigma[:, i]

        mean_statistic = stats.binned_statistic_2d(
            cluster_z,
            cluster_richness,
            cluster_DeltaSigma_at_radius,
            "mean",
            bins=[z_edges, richness_edges],
        ).statistic

        std_statistic = stats.binned_statistic_2d(
            cluster_z,
            cluster_richness,
            cluster_DeltaSigma_at_radius,
            "std",
            bins=[z_edges, richness_edges],
        ).statistic
        mean_DeltaSigma[:, :, i] = mean_statistic
        std_DeltaSigma[:, :, i] = std_statistic
    var_mean_DeltaSigma = std_DeltaSigma**2 / cluster_counts[..., None]
    # correlation matrix - the "large blocks" correspond to the $N_z$ redshift bins.
    # In each redshift bin are the $N_{\rm richness}$ richness bins.**
    covariance = np.diag(
        np.concatenate(
            (
                cluster_counts.flatten(),
                var_mean_logM.flatten(),
                var_mean_DeltaSigma.flatten(),
            )
        )
    )

    # Prepare the SACC file
    s_count = sacc.Sacc()
    bin_z_labels = []
    bin_richness_labels = []
    bin_radius_labels = []

    survey_name = "numcosmo_simulated_redshift_richness_deltasigma"
    s_count.add_tracer("survey", survey_name, area)

    for i, z_bin in enumerate(zip(z_edges[:-1], z_edges[1:])):
        lower, upper = z_bin
        bin_z_label = f"bin_z_{i}"
        s_count.add_tracer("bin_z", bin_z_label, lower, upper)
        bin_z_labels.append(bin_z_label)

    for i, richness_bin in enumerate(zip(richness_edges[:-1], richness_edges[1:])):
        lower, upper = richness_bin
        bin_richness_label = f"rich_{i}"
        s_count.add_tracer("bin_richness", bin_richness_label, lower, upper)
        bin_richness_labels.append(bin_richness_label)

    for i, radius_bin in enumerate(zip(radius_edges[:-1], radius_edges[1:])):
        radius_lower, radius_upper = radius_bin
        j = i + 2
        radius_center = np.mean(radius_edges[i:j])
        bin_radius_label = f"bin_radius_{i}"
        s_count.add_tracer(
            "bin_radius", bin_radius_label, radius_lower, radius_upper, radius_center
        )
        bin_radius_labels.append(bin_radius_label)

    #  pylint: disable-next=no-member
    cluster_count = sacc.standard_types.cluster_counts
    #  pylint: disable-next=no-member
    cluster_mean_log_mass = sacc.standard_types.cluster_mean_log_mass
    #  pylint: disable-next=no-member
    cluster_mean_DeltaSigma = sacc.standard_types.cluster_shear

    counts_and_edges = zip(
        cluster_counts.flatten(), itertools.product(bin_z_labels, bin_richness_labels)
    )

    mean_logM_and_edges = zip(
        mean_logM.flatten(), itertools.product(bin_z_labels, bin_richness_labels)
    )

    for counts, (bin_z_label, bin_richness_label) in counts_and_edges:
        s_count.add_data_point(
            cluster_count, (survey_name, bin_z_label, bin_richness_label), int(counts)
        )

    for bin_mean_logM, (bin_z_label, bin_richness_label) in mean_logM_and_edges:
        s_count.add_data_point(
            cluster_mean_log_mass,
            (survey_name, bin_z_label, bin_richness_label),
            bin_mean_logM,
        )
    for j, bin_z_label in enumerate(bin_z_labels):
        for k, bin_richness_label in enumerate(bin_richness_labels):
            for i, bin_radius_label in enumerate(bin_radius_labels):
                profile = mean_DeltaSigma[j][k][i]
                s_count.add_data_point(
                    cluster_mean_DeltaSigma,
                    (survey_name, bin_z_label, bin_richness_label, bin_radius_label),
                    profile,
                )
    # ### Then the add the covariance and save the file

    s_count.add_covariance(covariance)
    s_count.to_canonical_order()
    s_count.save_fits(
        "cluster_redshift_richness_deltasigma_sacc_data.fits", overwrite=True
    )


Ncm.cfg_init()
generate_sacc_file()
