"""Utilities for creating combinations of photo-z bins and measurements."""

from itertools import product, chain

import numpy as np

import firecrown.metadata_types as mdt


def make_all_photoz_bin_combinations(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
) -> list[mdt.TwoPointXY]:
    """Extract the two-point function metadata from a sacc file."""
    bin_combinations = [
        mdt.TwoPointXY(
            x=igz1, y=igz2, x_measurement=x_measurement, y_measurement=y_measurement
        )
        for igz1, igz2 in product(inferred_galaxy_zdists, repeat=2)
        for x_measurement, y_measurement in product(
            igz1.measurements, igz2.measurements
        )
        if mdt.measurement_is_compatible(x_measurement, y_measurement)
    ]

    return bin_combinations


def make_all_photoz_bin_combinations_with_cmb(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
    cmb_tracer_name: str = "cmb_convergence",
    include_cmb_auto: bool = False,
) -> list[mdt.TwoPointXY]:
    """Create all galaxy combinations plus mdt.CMB-galaxy cross-correlations.

    :param inferred_galaxy_zdists: List of galaxy redshift bins
    :param cmb_tracer_name: Name of the mdt.CMB tracer
    :param include_cmb_auto: Whether to include mdt.CMB auto-correlation
        (default: False)
    :return: List of all XY combinations including mdt.CMB-galaxy crosses
    """
    # Get all galaxy-galaxy combinations first
    galaxy_combinations = make_all_photoz_bin_combinations(inferred_galaxy_zdists)
    all_combinations = galaxy_combinations + make_cmb_galaxy_combinations_only(
        inferred_galaxy_zdists, cmb_tracer_name, include_cmb_auto
    )

    return all_combinations


def make_cmb_galaxy_combinations_only(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
    cmb_tracer_name: str = "cmb_convergence",
    include_cmb_auto: bool = False,
) -> list[mdt.TwoPointXY]:
    """Create only mdt.CMB-galaxy cross-correlations.

    :param inferred_galaxy_zdists: List of galaxy redshift bins
    :param cmb_tracer_name: Name of the mdt.CMB tracer
    :return: List of mdt.CMB-galaxy cross-correlation XY combinations only
    """
    # Create a mock mdt.CMB "bin"
    cmb_bin = mdt.InferredGalaxyZDist(
        bin_name=cmb_tracer_name,
        z=np.array([1100.0]),
        dndz=np.array([1.0]),
        measurements={mdt.CMB.CONVERGENCE},
        type_source=mdt.TypeSource.DEFAULT,
    )

    cmb_galaxy_combinations = []

    for galaxy_bin in inferred_galaxy_zdists:
        galaxy_bin_type = list(product([galaxy_bin], list(galaxy_bin.measurements)))
        cmb_bin_type = list(product([cmb_bin], list(cmb_bin.measurements)))
        for (x, m1), (y, m2) in chain(
            product(galaxy_bin_type, cmb_bin_type),
            product(cmb_bin_type, galaxy_bin_type),
        ):
            if mdt.measurement_is_compatible(m1, m2):
                # mdt.CMB-galaxy cross-correlation
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(x=x, y=y, x_measurement=m1, y_measurement=m2)
                )

    # Optionally include mdt.CMB auto-correlation
    if include_cmb_auto:
        cmb_galaxy_combinations.append(
            mdt.TwoPointXY(
                x=cmb_bin,
                y=cmb_bin,
                x_measurement=mdt.CMB.CONVERGENCE,
                y_measurement=mdt.CMB.CONVERGENCE,
            )
        )

    return cmb_galaxy_combinations
