#!/usr/bin/env python

"""
Generate a single SACC file.

The file contains:
 - cluster_counts
 - cluster_mean_log_mass
 - cluster_delta_sigma
 - cluster_shear  (reduced tangential shear, g_t)

All computed from the same simulated data.
"""

import os
from typing import Tuple
import math
import itertools
import numpy as np

from numcosmo_py import Nc, Ncm
from astropy.io import fits
from astropy.table import Table
from scipy import stats
import sacc
import pyccl as ccl

os.environ["CLMM_MODELING_BACKEND"] = "ccl"
# pylint: disable=C0413
import clmm  # noqa: E402
from clmm import Cosmology  # noqa: E402
from clmm.utils.beta_lens import (  # noqa: E402
    compute_beta_s_mean_from_distribution,
    compute_beta_s_square_mean_from_distribution,
)


####################################################################
# Cosmology
####################################################################


def generate_cosmo(
    H0: float, Ob0: float, Odm0: float, n_s: float, sigma8: float
) -> Tuple[Nc.HICosmoDECpl, ccl.Cosmology]:
    """
    Generate a cosmology object with the given parameters.

    :param H0: Hubble constant in km/s/Mpc.
    :param Ob0: Baryon density parameter.
    :param Odm0: Dark matter density parameter.
    :param n_s: Scalar spectral index.
    :param sigma8: Amplitude of matter fluctuations on scales of 8 Mpc/h.
    :return: A tuple containing:
        - Nc.HICosmoDECpl cosmology object.
        - pyccl cosmology object.
    """
    cosmo = Nc.HICosmoDECpl()
    cosmo_ccl = ccl.Cosmology(
        Omega_b=Ob0, Omega_c=Odm0, sigma8=sigma8, w0=-1, wa=0, h=H0 / 100.0, n_s=n_s
    )

    reion = Nc.HIReionCamb.new()  # pylint: disable=no-value-for-parameter
    prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
    cosmo.add_submodel(reion)
    cosmo.add_submodel(prim)

    tf = Nc.TransferFuncEH.new()  # pylint: disable=no-value-for-parameter
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

    old_amplitude = math.exp(prim.props.ln10e10ASA)
    prim.props.ln10e10ASA = math.log((sigma8 / cosmo.sigma8(psf)) ** 2 * old_amplitude)

    return cosmo, cosmo_ccl


####################################################################
# Simulate clusters (same for all statistics)
####################################################################


def generate_cluster_data(
    cosmo: Nc.HICosmoDECpl,
    area: float,
    M0: float,
    z0: float,
    lnRl: float,
    lnRu: float,
    zl: float,
    zu: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate cluster data including redshift, richness, and log mass.

    :param cosmo: Nc.HICosmoDECpl cosmology object.
    :param area: Survey area in square degrees.
    :param M0: Characteristic mass at z=0 in M_sun/h.
    :param z0: Characteristic redshift for mass evolution.
    :param lnRl: Minimum natural log of cluster richness.
    :param lnRu: Maximum natural log of cluster richness.
    :param zl: Minimum redshift for clusters.
    :param zu: Maximum redshift for clusters.
    :return: Tuple containing:
        - cluster_z: Redshift array.
        - cluster_richness: Richness array.
        - cluster_logM: Log mass array.
    """
    cluster_z = Nc.ClusterRedshiftNodist(z_max=zu, z_min=zl)
    cluster_m = Nc.ClusterMassAscaso(
        M0=M0, z0=z0, lnRichness_min=lnRl, lnRichness_max=lnRu
    )

    # Set mass parameters based on arXiv 1904.07524v2
    cluster_m.param_set_by_name("mup0", 3.19)
    cluster_m.param_set_by_name("mup1", 2 / np.log(10))
    cluster_m.param_set_by_name("mup2", -0.7 / np.log(10))
    cluster_m.param_set_by_name("sigmap0", 0.33)
    cluster_m.param_set_by_name("sigmap1", -0.08 / np.log(10))
    cluster_m.param_set_by_name("sigmap2", 0 / np.log(10))

    # Halo Mass Function
    mulf = Nc.MultiplicityFuncTinker.new()  # pylint: disable=no-value-for-parameter
    mulf.set_linear_interp(True)
    mulf.set_mdef(Nc.MultiplicityFuncMassDef.CRITICAL)
    mulf.set_Delta(200)

    hmf = Nc.HaloMassFunction.new(
        Nc.Distance.new(2.0),
        Ncm.PowspecFilter.new(
            Nc.PowspecMLTransfer.new(
                Nc.TransferFuncEH.new()  # pylint: disable=no-value-for-parameter
            ),
            Ncm.PowspecFilterType.TOPHAT,
        ),
        mulf,
    )

    hmf.set_area_sd(area)
    ca = Nc.ClusterAbundance.new(hmf, None)

    # Prepare cluster abundance object
    ca.prepare(cosmo, cluster_z, cluster_m)
    mset = Ncm.MSet.new_array([cosmo, cluster_z, cluster_m])

    rng = Ncm.RNG.seeded_new(None, 2)
    ncount = Nc.DataClusterNCount.new(
        ca, "NcClusterRedshiftNodist", "NcClusterMassAscaso"
    )
    ncount.init_from_sampling(mset, area * ((np.pi / 180) ** 2), rng)

    # Extract data
    ncount.catalog_save("ncount_rich.fits", True)
    ncdata_fits = fits.open("ncount_rich.fits")
    #  pylint: disable-next=no-member
    ncdata_data = ncdata_fits[1].data
    ncdata_Table = Table(ncdata_data)

    data_table = ncdata_Table[ncdata_Table["LNM_OBS"] > 2]
    cluster_z = data_table["Z_OBS"]
    cluster_richness = data_table["LNM_OBS"] / np.log(10.0)
    cluster_logM = data_table["LNM_TRUE"] / np.log(10.0)

    return cluster_z, cluster_richness, cluster_logM


####################################################################
# Compute DeltaSigma and reduced shear, consistently binned
####################################################################


def compute_profiles(N_z: int, N_rich: int, cosmo_ccl, z, richness, logM):
    """Compute lensing profile, Delta Sigma and g_t."""
    # binning
    counts, z_edges, r_edges, _ = stats.binned_statistic_2d(
        z, richness, logM, "count", bins=[N_z, N_rich]
    )

    mean_logM = stats.binned_statistic_2d(
        z, richness, logM, "mean", bins=[z_edges, r_edges]
    ).statistic

    std_logM = stats.binned_statistic_2d(
        z, richness, logM, "std", bins=[z_edges, r_edges]
    ).statistic

    var_logM = std_logM**2 / counts

    # CLMM modeling
    cosmo_clmm = Cosmology(be_cosmo=cosmo_ccl)

    moo = clmm.Modeling(massdef="critical", delta_mdef=200, halo_profile_model="nfw")
    moo.set_cosmo(cosmo_clmm)
    # assuming the same concentration for all masses. Not realistic,
    # but avoid having to call a mass-concentration relation.
    moo.set_concentration(4)

    radius_edges = clmm.make_bins(
        0.3, 6.0, nbins=6, method="evenlog10width"
    )  # 6 radial bins log-spaced between 0.3 and 6 Mpc

    radius_centers = [
        np.mean(radius_edges[i: i + 2]) for i in range(len(radius_edges) - 1)
    ]

    all_dsigma_list = []
    all_gt_list = []

    for zc, lM in zip(z, logM):
        mass = 10**lM
        moo.set_mass(mass)

        # DeltaSigma
        ds = moo.eval_excess_surface_density(radius_centers, zc)

        # reduced shear g_t
        beta = compute_beta_s_mean_from_distribution(
            z_cl=zc, z_inf=10, cosmo=moo.cosmo, zmax=5.0
        )
        beta2 = compute_beta_s_square_mean_from_distribution(
            z_cl=zc, z_inf=10, cosmo=moo.cosmo, zmax=5.0
        )

        gt = moo.eval_reduced_tangential_shear(
            radius_centers, zc, (beta, beta2), z_src_info="beta", approx="order1"
        )

        all_dsigma_list.append(ds)
        all_gt_list.append(gt)

    all_dsigma = np.array(all_dsigma_list)
    all_gt = np.array(all_gt_list)

    nb = len(radius_centers)

    mean_dsigma: np.ndarray = np.zeros((N_z, N_rich, nb))
    mean_gt: np.ndarray = np.zeros((N_z, N_rich, nb))

    var_dsigma: np.ndarray = np.zeros((N_z, N_rich, nb))
    var_gt: np.ndarray = np.zeros((N_z, N_rich, nb))

    for ri in range(nb):
        ds_i = all_dsigma[:, ri]
        gt_i = all_gt[:, ri]

        mean_dsigma[:, :, ri] = stats.binned_statistic_2d(
            z, richness, ds_i, "mean", bins=[z_edges, r_edges]
        ).statistic

        std_dsigma = stats.binned_statistic_2d(
            z, richness, ds_i, "std", bins=[z_edges, r_edges]
        ).statistic

        var_dsigma[:, :, ri] = std_dsigma**2 / counts

        mean_gt[:, :, ri] = stats.binned_statistic_2d(
            z, richness, gt_i, "mean", bins=[z_edges, r_edges]
        ).statistic

        std_gt = stats.binned_statistic_2d(
            z, richness, gt_i, "std", bins=[z_edges, r_edges]
        ).statistic

        var_gt[:, :, ri] = std_gt**2 / counts

    # build diagonal covariance
    cov = np.diag(
        np.concatenate(
            [
                counts.flatten(),
                var_logM.flatten(),
                var_dsigma.flatten(),
                var_gt.flatten(),
            ]
        )
    )

    return (
        counts,
        mean_logM,
        mean_dsigma,
        mean_gt,
        z_edges,
        r_edges,
        radius_edges,
        cov,
    )


####################################################################
# Build SACC file
####################################################################


def build_tracers(s, survey, z_edges, r_edges, R_edges):
    """Add all tracers to SACC and return label lists."""
    z_labels, r_labels, R_labels = [], [], []

    # Redshift bins
    for i, (lo, hi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        name = f"bin_z_{i}"
        s.add_tracer("bin_z", name, lo, hi)
        z_labels.append(name)

    # Richness bins
    for i, (lo, hi) in enumerate(zip(r_edges[:-1], r_edges[1:])):
        name = f"bin_rich_{i}"
        s.add_tracer("bin_richness", name, lo, hi)
        r_labels.append(name)

    # Radius bins
    R_centers = [np.mean(R_edges[i: i + 2]) for i in range(len(R_edges) - 1)]
    for i, (lo, hi) in enumerate(zip(R_edges[:-1], R_edges[1:])):
        name = f"bin_radius_{i}"
        s.add_tracer("bin_radius", name, float(lo), float(hi), float(R_centers[i]))
        R_labels.append(name)

    return z_labels, r_labels, R_labels


def fill_counts_and_mass(
    s: sacc.Sacc,
    survey: str,
    counts: np.ndarray,
    mean_logM: np.ndarray,
    z_labels: list[str],
    r_labels: list[str],
    t_counts: str,
    t_logmass: str,
) -> None:
    """Fill SACC object with cluster counts and mean log mass."""
    counts_and_edges = zip(counts.flatten(), itertools.product(z_labels, r_labels))

    mean_logM_and_edges = zip(
        mean_logM.flatten(), itertools.product(z_labels, r_labels)
    )

    for counts, (bin_z_label, bin_richness_label) in counts_and_edges:
        s.add_data_point(
            t_counts, (survey, bin_z_label, bin_richness_label), int(counts)
        )

    for bin_mean_logM, (bin_z_label, bin_richness_label) in mean_logM_and_edges:
        s.add_data_point(
            t_logmass,
            (survey, bin_z_label, bin_richness_label),
            bin_mean_logM,
        )


def fill_profiles(
    s: sacc.Sacc,
    survey: str,
    mean_dsigma: np.ndarray,
    mean_gt: np.ndarray,
    z_labels: list[str],
    r_labels: list[str],
    R_labels: list[str],
    t_dsigma: str,
    t_shear: str,
) -> None:
    """Fill SACC object with DeltaSigma and reduced shear profiles."""
    for zi, zlab in enumerate(z_labels):
        for ri, rlab in enumerate(r_labels):
            for iR, Rlab in enumerate(R_labels):
                s.add_data_point(
                    t_dsigma,
                    (str(survey), str(zlab), str(rlab), str(Rlab)),
                    float(mean_dsigma[zi, ri, iR]),
                )

    for zi, zlab in enumerate(z_labels):
        for ri, rlab in enumerate(r_labels):
            for iR, Rlab in enumerate(R_labels):
                s.add_data_point(
                    t_shear,
                    (str(survey), str(zlab), str(rlab), str(Rlab)),
                    float(mean_gt[zi, ri, iR]),
                )


def generate_sacc_file():
    """Generate the full SACC file for the simulated cluster data."""
    # --- config ---
    area = 439.78986
    H0 = 71.0
    Ob0 = 0.0448
    Odm0 = 0.22
    n_s = 0.963
    sigma8 = 0.8
    M0 = 3e14 / 0.71
    z0 = 0.6
    lnRl, lnRu = 0.0, 5.0
    zl, zu = 0.2, 0.65
    N_z = 4
    N_richness = 5

    # --- generate data ---
    cosmo, cosmo_ccl = generate_cosmo(H0, Ob0, Odm0, n_s, sigma8)
    z, richness, logM = generate_cluster_data(cosmo, area, M0, z0, lnRl, lnRu, zl, zu)

    (counts, mean_logM, mean_dsigma, mean_gt, z_edges, r_edges, R_edges, cov) = (
        compute_profiles(N_z, N_richness, cosmo_ccl, z, richness, logM)
    )

    # --- SACC setup ---
    s = sacc.Sacc()
    survey = "numcosmo_sim_red_richness_shear_dsigma"
    s.add_tracer("survey", survey, area)

    # Add tracers
    (z_labels, r_labels, R_labels) = build_tracers(s, survey, z_edges, r_edges, R_edges)

    # Data types
    t_counts = sacc.standard_types.cluster_counts
    t_logmass = sacc.standard_types.cluster_mean_log_mass
    t_dsigma = sacc.standard_types.cluster_delta_sigma
    t_shear = sacc.standard_types.cluster_shear

    # Fill data blocks
    fill_counts_and_mass(
        s, survey, counts, mean_logM, z_labels, r_labels, t_counts, t_logmass
    )
    fill_profiles(
        s, survey, mean_dsigma, mean_gt, z_labels, r_labels, R_labels, t_dsigma, t_shear
    )

    # Covariance & save
    s.add_covariance(cov)
    s.to_canonical_order()
    s.save_fits("cluster_redshift_richness_shear_dsigma_sacc.fits", overwrite=True)


####################################################################

Ncm.cfg_init()
generate_sacc_file()
