#!/usr/bin/env python

"""Defines a function to generate a SACC file for cluster number counts."""

# # Cluster count-only SACC file creation
#
# This notebook examplifies the creation of a SACC file for cluster count, using
# NumCosmo facilities to simulate cluster data.

import itertools

from numcosmo_py import Ncm

import sacc


def convert_binned_profile_to_sacc(
    cluster_counts,
    mean_DeltaSigma,
    covariance,
    z_edges,
    richness_edges,
    radius_edges,
    radius_centers,
    area,
    survey_name="numcosmo_simulated_redshift_richness",
) -> sacc.Sacc:
    """Function to generate and store SACC data for cluste counts."""
    N_z = len(z_edges) - 1
    N_richness = len(richness_edges) - 1
    N_radius = len(radius_edges) - 1

    # Prepare the SACC file
    s_count = sacc.Sacc()
    bin_z_labels = []
    bin_richness_labels = []
    bin_radius_labels = []

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

    for i, (radius_lower, radius_upper, radius_center) in enumerate(
        zip(radius_edges[:-1], radius_edges[1:], radius_centers)
    ):
        bin_radius_label = f"bin_radius_{i}"
        s_count.add_tracer(
            "bin_radius", bin_radius_label, radius_lower, radius_upper, radius_center
        )
        bin_radius_labels.append(bin_radius_label)

    #  pylint: disable-next=no-member
    type_cluster_counts = sacc.standard_types.cluster_counts
    #  pylint: disable-next=no-member
    type_cluster_DeltaSigma = sacc.standard_types.cluster_shear

    for counts, (bin_z_label, bin_richness_label) in zip(
        cluster_counts.flatten(), itertools.product(bin_z_labels, bin_richness_labels)
    ):
        s_count.add_data_point(
            type_cluster_counts,
            (survey_name, bin_z_label, bin_richness_label),
            int(counts),
        )

    for profile, (bin_z_label, bin_richness_label) in zip(
        mean_DeltaSigma.reshape(N_richness * N_z, N_radius),
        itertools.product(bin_z_labels, bin_richness_labels),
    ):
        for profile_value, bin_radius_label in zip(profile, bin_radius_labels):
            s_count.add_data_point(
                type_cluster_DeltaSigma,
                (survey_name, bin_z_label, bin_richness_label, bin_radius_label),
                profile_value,
            )

    # ### Then the add the covariance and save the file

    s_count.add_covariance(covariance)
    s_count.to_canonical_order()
    return s_count


if __name__ == "__main__":
    Ncm.cfg_init()
