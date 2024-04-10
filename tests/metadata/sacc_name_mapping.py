"""Module containing testing data for translation of SACC names."""

from firecrown.metadata.two_point import (
    ClusterMeasuredType,
    CMBMeasuredType,
    GalaxyMeasuredType,
)


mappings = [
    (
        "clusterGalaxy_densityShear_cl_e",
        "ell",
        (ClusterMeasuredType.COUNTS, GalaxyMeasuredType.SHEAR_E),
    ),
    (
        "clusterGalaxy_densityShear_xi_t",
        "theta",
        (ClusterMeasuredType.COUNTS, GalaxyMeasuredType.SHEAR_T),
    ),
    (
        "clusterGalaxy_density_cl",
        "ell",
        (ClusterMeasuredType.COUNTS, GalaxyMeasuredType.COUNTS),
    ),
    (
        "clusterGalaxy_density_xi",
        "theta",
        (ClusterMeasuredType.COUNTS, GalaxyMeasuredType.COUNTS),
    ),
    (
        "cluster_density_cl",
        "ell",
        (ClusterMeasuredType.COUNTS, ClusterMeasuredType.COUNTS),
    ),
    (
        "cluster_density_xi",
        "theta",
        (ClusterMeasuredType.COUNTS, ClusterMeasuredType.COUNTS),
    ),
    (
        "cmbCluster_convergenceDensity_cl",
        "ell",
        (CMBMeasuredType.CONVERGENCE, ClusterMeasuredType.COUNTS),
    ),
    (
        "cmbCluster_convergenceDensity_xi",
        "theta",
        (CMBMeasuredType.CONVERGENCE, ClusterMeasuredType.COUNTS),
    ),
    (
        "cmbGalaxy_convergenceDensity_cl",
        "ell",
        (CMBMeasuredType.CONVERGENCE, GalaxyMeasuredType.COUNTS),
    ),
    (
        "cmbGalaxy_convergenceDensity_xi",
        "theta",
        (CMBMeasuredType.CONVERGENCE, GalaxyMeasuredType.COUNTS),
    ),
    (
        "cmbGalaxy_convergenceShear_cl_e",
        "ell",
        (CMBMeasuredType.CONVERGENCE, GalaxyMeasuredType.SHEAR_E),
    ),
    (
        "cmbGalaxy_convergenceShear_xi_t",
        "theta",
        (CMBMeasuredType.CONVERGENCE, GalaxyMeasuredType.SHEAR_T),
    ),
    (
        "cmb_convergence_cl",
        "ell",
        (CMBMeasuredType.CONVERGENCE, CMBMeasuredType.CONVERGENCE),
    ),
    (
        "cmb_convergence_xi",
        "theta",
        (CMBMeasuredType.CONVERGENCE, CMBMeasuredType.CONVERGENCE),
    ),
    (
        "galaxy_density_cl",
        "ell",
        (GalaxyMeasuredType.COUNTS, GalaxyMeasuredType.COUNTS),
    ),
    (
        "galaxy_density_xi",
        "theta",
        (GalaxyMeasuredType.COUNTS, GalaxyMeasuredType.COUNTS),
    ),
    (
        "galaxy_shearDensity_cl_e",
        "ell",
        (GalaxyMeasuredType.SHEAR_E, GalaxyMeasuredType.COUNTS),
    ),
    (
        "galaxy_shearDensity_xi_t",
        "theta",
        (GalaxyMeasuredType.SHEAR_T, GalaxyMeasuredType.COUNTS),
    ),
    (
        "galaxy_shear_cl_ee",
        "ell",
        (GalaxyMeasuredType.SHEAR_E, GalaxyMeasuredType.SHEAR_E),
    ),
]
