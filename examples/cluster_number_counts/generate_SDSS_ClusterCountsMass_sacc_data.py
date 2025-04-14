#!/usr/bin/env python

"""Defines a function to generate a SACC file for cluster number counts."""

# Cluster SACC file creation for SDSS Count+Mean(logM) data vector
# in Costanzi et al. 2019, https://arxiv.org/pdf/1810.09456

import itertools
import numpy as np
from numcosmo_py import Ncm
import sacc


def generate_SDSSCL_sacc_file() -> None:
    """
    Generate a SACC file with SDSS cluster number counts and Mean(logM).

    According to SDSS Costanzai et al. 2019, arxiv 1810.09456.
    Output: will save a fits file to disk.
    """
    # settting cluster data vector
    # according to this Costanzi et al. 2019 arxiv 1810.09456
    area = 10263.037032448827
    z_edges = np.array([0.1, 0.3])
    richness_edges = np.log10(np.array([20, 27.9, 37.6, 50.3, 69.3, 140]))
    cluster_counts = np.array([3711, 1788, 978, 476, 223])
    mean_logM = np.array([14.111, 14.263, 14.380, 14.609, 14.928])
    std_logM1 = np.array([0.024, 0.030, 0.033, 0.036, 0.029])
    std_logM2 = np.array([0.026, 0.024, 0.026, 0.028, 0.036])
    std_counts = np.array([100, 61, 41, 27, 18])
    var_mean_logM = std_logM1**2 + std_logM2**2
    var_counts = std_counts**2

    # ** Correlation matrix - the "large blocks" correspond to the $N_z$ redshift bins.
    # In each redshift bin are the $N_{\rm richness}$ richness bins.**

    covariance = np.diag(
        np.concatenate((var_counts.flatten(), var_mean_logM.flatten()))
    )

    s_count = sacc.Sacc()
    bin_z_labels = []
    bin_richness_labels = []

    survey_name = "SDSSCluster_redshift_richness"
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

    #  pylint: disable-next=no-member
    cluster_count = sacc.standard_types.cluster_counts
    #  pylint: disable-next=no-member
    cluster_mean_log_mass = sacc.standard_types.cluster_mean_log_mass

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

    # ### Then the add the covariance and save the file

    s_count.add_covariance(covariance)
    s_count.to_canonical_order()
    s_count.save_fits("SDSSTEST_redshift_richness_sacc_data.fits", overwrite=True)


if __name__ == "__main__":
    Ncm.cfg_init()  # pylint: disable=no-value-for-parameter
    generate_SDSSCL_sacc_file()
