"""Utilities for creating combinations of photo-z bins and measurements."""

from itertools import combinations_with_replacement, product

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
        for igz1, igz2 in combinations_with_replacement(inferred_galaxy_zdists, 2)
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

    # Create a mock mdt.CMB "bin" for cross-correlations
    cmb_bin = mdt.InferredGalaxyZDist(
        bin_name=cmb_tracer_name,
        z=np.array([1100.0]),  # mdt.CMB redshift
        dndz=np.array([1.0]),  # Unity normalization
        measurements={mdt.CMB.CONVERGENCE},
        type_source=mdt.TypeSource.DEFAULT,
    )

    # Create mdt.CMB-galaxy cross-correlations only
    cmb_galaxy_combinations = []

    for galaxy_bin in inferred_galaxy_zdists:
        for galaxy_measurement in galaxy_bin.measurements:
            # Only create cross-correlations that are physically meaningful
            if mdt.measurement_is_compatible(mdt.CMB.CONVERGENCE, galaxy_measurement):
                # mdt.CMB-galaxy cross-correlation
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(
                        x=cmb_bin,
                        y=galaxy_bin,
                        x_measurement=mdt.CMB.CONVERGENCE,
                        y_measurement=galaxy_measurement,
                    )
                )

                # Galaxy-mdt.CMB cross-correlation (symmetric)
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(
                        x=galaxy_bin,
                        y=cmb_bin,
                        x_measurement=galaxy_measurement,
                        y_measurement=mdt.CMB.CONVERGENCE,
                    )
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

    return galaxy_combinations + cmb_galaxy_combinations


def make_cmb_galaxy_combinations_only(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
    cmb_tracer_name: str = "cmb_convergence",
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
        for galaxy_measurement in galaxy_bin.measurements:
            if mdt.measurement_is_compatible(mdt.CMB.CONVERGENCE, galaxy_measurement):
                # mdt.CMB-galaxy cross-correlation
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(
                        x=cmb_bin,
                        y=galaxy_bin,
                        x_measurement=mdt.CMB.CONVERGENCE,
                        y_measurement=galaxy_measurement,
                    )
                )

                # Galaxy-mdt.CMB cross-correlation (symmetric)
                cmb_galaxy_combinations.append(
                    mdt.TwoPointXY(
                        x=galaxy_bin,
                        y=cmb_bin,
                        x_measurement=galaxy_measurement,
                        y_measurement=mdt.CMB.CONVERGENCE,
                    )
                )

    return cmb_galaxy_combinations


def make_all_bin_rule_combinations(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
    bin_rule: mdt.BinRule,
) -> list[mdt.TwoPointXY]:
    """Extract the two-point function metadata from a sacc file."""
    bin_combinations = [
        mdt.TwoPointXY(
            x=igz1, y=igz2, x_measurement=x_measurement, y_measurement=y_measurement
        )
        for igz1, igz2 in combinations_with_replacement(inferred_galaxy_zdists, 2)
        for x_measurement, y_measurement in product(
            igz1.measurements, igz2.measurements
        )
        if mdt.measurement_is_compatible(x_measurement, y_measurement)
        and (bin_rule.keep((igz1, igz2), (x_measurement, y_measurement)))
    ]

    return bin_combinations
