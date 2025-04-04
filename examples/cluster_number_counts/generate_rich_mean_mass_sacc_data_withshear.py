#!/usr/bin/env python

"""Function to generate a SACC file for cluster number counts and cluster DeltaSigma."""
import os
from typing import Tuple

import math
import itertools
import numpy as np
from numcosmo_py import Nc, Ncm
from astropy.table import Table
from astropy.io import fits
from scipy import stats
import sacc
import pyccl as ccl

os.environ["CLMM_MODELING_BACKEND"] = "ccl"
# pylint: disable=C0413
import clmm  # noqa: E402
from clmm import Cosmology  # noqa: E402


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


def compute_abundance_deltasigma_statistic(
    N_richness: float, N_z: float, cosmo_ccl, cluster_z, cluster_richness, cluster_logM
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Computes abundance statistics and DeltaSigma for clusters.

    :param N_richness: Number of richness bins.
    :param N_z: Number of redshift bins.
    :param cosmo_ccl: pyCCL cosmology object.
    :param cluster_z: Array of cluster redshifts.
    :param cluster_richness: Array of cluster richness values.
    :param cluster_logM: Array of cluster log mass values.
    :return: (Cluster counts, mean log mass, mean DeltaSigma,
            redshift bin edges, richness bin edges, radius bin edges, covariance matrix)
    """
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
    cosmo_clmm._init_from_cosmo(cosmo_ccl)  # pylint: disable=protected-access
    moo = clmm.Modeling(massdef="critical", delta_mdef=200, halo_profile_model="nfw")
    moo.set_cosmo(cosmo_clmm)
    # assuming the same concentration for all masses. Not realistic,
    # but avoid having to call a mass-concentration relation.
    moo.set_concentration(4)

    radius_edges = clmm.make_bins(
        0.3, 6.0, nbins=6, method="evenlog10width"
    )  # 6 radial bins log-spaced between 0.3 and 6 Mpc

    radius_centers = []
    for i in range(len(radius_edges) - 1):
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
    for i in range(len(radius_edges) - 1):
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
    return (
        cluster_counts,
        mean_logM,
        mean_DeltaSigma,
        z_edges,
        richness_edges,
        radius_edges,
        covariance,
    )


def generate_sacc_file() -> None:
    """Generate and save a SACC file for cluster number counts and DeltaSigma."""
    # Define parameter values explicitly
    area = 439.78986
    H0 = 71.0
    Ob0 = 0.0448
    Odm0 = 0.22
    n_s = 0.963
    sigma8 = 0.8
    M0 = 3.0e14 / 0.71
    z0 = 0.6
    lnRl = 0.0
    lnRu = 5.0
    zl = 0.2
    zu = 0.65
    N_richness = 5
    N_z = 4

    # Generate cosmology and cluster data
    cosmo, cosmo_ccl = generate_cosmo(H0, Ob0, Odm0, n_s, sigma8)
    cluster_z, cluster_richness, cluster_logM = generate_cluster_data(
        cosmo, area, M0, z0, lnRl, lnRu, zl, zu
    )
    (
        cluster_counts,
        mean_logM,
        mean_DeltaSigma,
        z_edges,
        richness_edges,
        radius_edges,
        covariance,
    ) = compute_abundance_deltasigma_statistic(
        N_richness, N_z, cosmo_ccl, cluster_z, cluster_richness, cluster_logM
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


Ncm.cfg_init()  # pylint: disable=no-value-for-parameter
generate_sacc_file()
