#!/usr/bin/env python

"""Defines a function to generate a SACC file for cluster number counts."""

# # Cluster count-only SACC file creation
#
# This notebook examplifies the creation of a SACC file for cluster count, using
# NumCosmo facilities to simulate cluster data.

import math
import itertools

import numpy as np

from numcosmo_py import Nc, Ncm

from astropy.table import Table

from astropy.io import fits
from scipy import stats
import sacc


def setup_cosmology(
    H0: float, Ob0: float, Odm0: float, n_s: float, sigma8: float
) -> Nc.HICosmoDEXcdm:
    """
    Set up the cosmological model with specified parameters.

    :param H0: Hubble constant in km/s/Mpc.
    :param Ob0: Baryon density parameter.
    :param Odm0: Dark matter density parameter.
    :param n_s: Scalar spectral index.
    :param sigma8: The amplitude of matter fluctuations on scales of 8 Mpc/h.
    :return: Configured cosmology object.
    """
    cosmo = Nc.HICosmoDEXcdm()
    reion = Nc.HIReionCamb.new()
    prim = Nc.HIPrimPowerLaw.new()

    cosmo.add_submodel(reion)
    cosmo.add_submodel(prim)

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

    return cosmo


def setup_cluster_models(
    z_min: float, z_max: float, M0: float, z0: float, lnRl: float, lnRu: float
) -> tuple:
    """
    Set up cluster redshift and mass models.

    :param z_min: Minimum redshift for clusters.
    :param z_max: Maximum redshift for clusters.
    :param M0: Characteristic mass at z=0 in units of M_sun/h.
    :param z0: Characteristic redshift for mass evolution.
    :param lnRl: Minimum natural log of cluster richness.
    :param lnRu: Maximum natural log of cluster richness.
    :return: Tuple of cluster redshift and mass objects.
    """
    cluster_z = Nc.ClusterRedshiftNodist(z_max=z_max, z_min=z_min)
    cluster_m = Nc.ClusterMassAscaso(
        M0=M0, z0=z0, lnRichness_min=lnRl, lnRichness_max=lnRu
    )

    # Setting parameters for the mass function
    for param, value in [
        ("mup0", 3.19),
        ("mup1", 2 / np.log(10)),
        ("mup2", -0.7 / np.log(10)),
        ("sigmap0", 0.33),
        ("sigmap1", -0.08 / np.log(10)),
        ("sigmap2", 0 / np.log(10)),
    ]:
        cluster_m.param_set_by_name(param, value)

    return cluster_z, cluster_m


def generate_cluster_data(
    cosmo: Nc.HICosmoDEXcdm,
    cluster_z: Nc.ClusterRedshiftNodist,
    cluster_m: Nc.ClusterMassAscaso,
    area: float,
) -> Table:
    """
    Generate cluster data based on the cosmological and cluster models.

    :param cosmo: Cosmology object from setup_cosmology.
    :param cluster_z: Cluster redshift model.
    :param cluster_m: Cluster mass model.
    :param area: Survey area in square degrees.
    :return: Astropy Table with cluster data.
    """
    mulf = Nc.MultiplicityFuncTinker.new()
    mulf.set_linear_interp(True)
    mulf.set_mdef(Nc.MultiplicityFuncMassDef.MEAN)
    mulf.set_Delta(200)

    hmf = Nc.HaloMassFunction.new(
        Nc.Distance.new(2.0),
        Ncm.PowspecFilter.new(
            Nc.PowspecMLTransfer.new(Nc.TransferFuncEH.new()),
            Ncm.PowspecFilterType.TOPHAT,
        ),
        mulf,
    )
    hmf.set_area_sd(area)

    ca = Nc.ClusterAbundance.new(hmf, None)
    ncount = Nc.DataClusterNCount.new(
        ca, "NcClusterRedshiftNodist", "NcClusterMassAscaso"
    )
    ca.prepare(cosmo, cluster_z, cluster_m)
    mset = Ncm.MSet.new_array([cosmo, cluster_z, cluster_m])

    rng = Ncm.RNG.seeded_new(None, 32)
    ncount.init_from_sampling(mset, area * ((np.pi / 180) ** 2), rng)

    ncount.catalog_save("ncount_rich.fits", True)
    ncdata_fits = fits.open("ncount_rich.fits")
    ncdata_data = ncdata_fits[1].data  # pylint: disable=no-member
    return Table(ncdata_data)


def process_data_for_sacc(data_table: Table) -> tuple:
    """
    Process cluster data for SACC file generation.

    :param data_table: Astropy Table containing raw cluster data.
    :return: Tuple of processed data ready for SACC format.
    """
    filtered_data = data_table[data_table["LNM_OBS"] > 2]
    cluster_z = filtered_data["Z_OBS"]
    cluster_lnm = filtered_data["LNM_OBS"]
    cluster_richness = cluster_lnm / np.log(10.0)
    cluster_logM = filtered_data["LNM_TRUE"] / np.log(10.0)

    N_richness = 5
    N_z = 4
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
    covariance = np.diag(
        np.concatenate((cluster_counts.flatten(), var_mean_logM.flatten()))
    )

    return cluster_counts, z_edges, richness_edges, mean_logM, covariance


def generate_sacc_file() -> None:
    """Generate a SACC file for cluster number counts."""
    H0, Ob0, Odm0, n_s, sigma8 = 71.0, 0.0448, 0.22, 0.963, 0.8
    area, lnRl, lnRu, zl, zu = 439.78986, 0.0, 5.0, 0.2, 0.65
    M0, z0 = 3.0e14 / 0.71, 0.6

    cosmo = setup_cosmology(H0, Ob0, Odm0, n_s, sigma8)
    cluster_z, cluster_m = setup_cluster_models(zl, zu, M0, z0, lnRl, lnRu)
    data_table = generate_cluster_data(cosmo, cluster_z, cluster_m, area)

    cluster_counts, z_edges, richness_edges, mean_logM, covariance = (
        process_data_for_sacc(data_table)
    )

    s_count = sacc.Sacc()
    survey_name = "numcosmo_simulated_redshift_richness"
    s_count.add_tracer("survey", survey_name, area)

    bin_z_labels = [f"bin_z_{i}" for i in range(len(z_edges) - 1)]
    bin_richness_labels = [f"rich_{i}" for i in range(len(richness_edges) - 1)]

    for i, (lower, upper) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        s_count.add_tracer("bin_z", bin_z_labels[i], lower, upper)

    for i, (lower, upper) in enumerate(zip(richness_edges[:-1], richness_edges[1:])):
        s_count.add_tracer("bin_richness", bin_richness_labels[i], lower, upper)

    counts_and_edges = zip(
        cluster_counts.flatten(), itertools.product(bin_z_labels, bin_richness_labels)
    )
    mean_logM_and_edges = zip(
        mean_logM.flatten(), itertools.product(bin_z_labels, bin_richness_labels)
    )

    for counts, (bin_z_label, bin_richness_label) in counts_and_edges:
        s_count.add_data_point(
            "cluster_counts",
            (survey_name, bin_z_label, bin_richness_label),
            int(counts),
        )

    for bin_mean_logM, (bin_z_label, bin_richness_label) in mean_logM_and_edges:
        s_count.add_data_point(
            "cluster_mean_log_mass",
            (survey_name, bin_z_label, bin_richness_label),
            bin_mean_logM,
        )

    s_count.add_covariance(covariance)
    s_count.to_canonical_order()
    s_count.save_fits("cluster_redshift_richness_sacc_data.fits", overwrite=True)


if __name__ == "__main__":
    Ncm.cfg_init()
    generate_sacc_file()
