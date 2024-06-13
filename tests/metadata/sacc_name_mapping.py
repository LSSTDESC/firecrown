"""Module containing testing data for translation of SACC names."""

from firecrown.metadata.two_point import (
    Clusters,
    CMB,
    Galaxies,
)


mappings = [
    (
        "clusterGalaxy_densityShear_cl_e",
        "ell",
        (Clusters.COUNTS, Galaxies.SHEAR_E),
    ),
    (
        "clusterGalaxy_densityShear_xi_t",
        "theta",
        (Clusters.COUNTS, Galaxies.SHEAR_T),
    ),
    (
        "clusterGalaxy_density_cl",
        "ell",
        (Clusters.COUNTS, Galaxies.COUNTS),
    ),
    (
        "clusterGalaxy_density_xi",
        "theta",
        (Clusters.COUNTS, Galaxies.COUNTS),
    ),
    (
        "cluster_density_cl",
        "ell",
        (Clusters.COUNTS, Clusters.COUNTS),
    ),
    (
        "cluster_density_xi",
        "theta",
        (Clusters.COUNTS, Clusters.COUNTS),
    ),
    (
        "cmbCluster_convergenceDensity_cl",
        "ell",
        (CMB.CONVERGENCE, Clusters.COUNTS),
    ),
    (
        "cmbCluster_convergenceDensity_xi",
        "theta",
        (CMB.CONVERGENCE, Clusters.COUNTS),
    ),
    (
        "cmbGalaxy_convergenceDensity_cl",
        "ell",
        (CMB.CONVERGENCE, Galaxies.COUNTS),
    ),
    (
        "cmbGalaxy_convergenceDensity_xi",
        "theta",
        (CMB.CONVERGENCE, Galaxies.COUNTS),
    ),
    (
        "cmbGalaxy_convergenceShear_cl_e",
        "ell",
        (CMB.CONVERGENCE, Galaxies.SHEAR_E),
    ),
    (
        "cmbGalaxy_convergenceShear_xi_t",
        "theta",
        (CMB.CONVERGENCE, Galaxies.SHEAR_T),
    ),
    (
        "cmb_convergence_cl",
        "ell",
        (CMB.CONVERGENCE, CMB.CONVERGENCE),
    ),
    (
        "cmb_convergence_xi",
        "theta",
        (CMB.CONVERGENCE, CMB.CONVERGENCE),
    ),
    (
        "galaxy_density_cl",
        "ell",
        (Galaxies.COUNTS, Galaxies.COUNTS),
    ),
    (
        "galaxy_density_xi",
        "theta",
        (Galaxies.COUNTS, Galaxies.COUNTS),
    ),
    (
        "galaxy_shearDensity_cl_e",
        "ell",
        (Galaxies.SHEAR_E, Galaxies.COUNTS),
    ),
    (
        "galaxy_shearDensity_xi_t",
        "theta",
        (Galaxies.SHEAR_T, Galaxies.COUNTS),
    ),
    (
        "galaxy_shear_cl_ee",
        "ell",
        (Galaxies.SHEAR_E, Galaxies.SHEAR_E),
    ),
]
