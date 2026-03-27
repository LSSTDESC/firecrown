"""Utilities for creating combinations of tomographic bins and measurements.

This module provides functions to generate two-point correlation combinations from
tomographic redshift bins. It supports:
- All possible galaxy-galaxy correlations
- CMB-galaxy cross-correlations
- Filtered combinations based on bin pair selectors
"""

from itertools import product, chain

import firecrown.metadata_types as mdt


def _validate_list_of_inferred_galaxy_zdists(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
) -> None:
    """Validate that tomographic bin names are unique.

    :param inferred_galaxy_zdists: List of tomographic bins to validate.

    :raises ValueError: If any bin names appear more than once in the list.
    """
    bin_names_set = set()
    # Produce a list of duplicates
    bin_names = []
    for igz in inferred_galaxy_zdists:
        if igz.bin_name in bin_names_set:
            bin_names.append(igz.bin_name)
        else:
            bin_names_set.add(igz.bin_name)

    if bin_names:
        raise ValueError(
            f"Duplicate inferred galaxy z distribution bin names found: {bin_names}"
        )


def make_all_photoz_bin_combinations(
    inferred_galaxy_zdists: list[mdt.TomographicBin],
) -> list[mdt.TwoPointXY]:
    """Create all possible two-point correlation combinations for galaxy bins.

    This function generates all possible pairs of (bin, measurement) combinations,
    keeping only those where the measurements are compatible. For auto-correlations
    (same measurement type), only unique pairs are kept to avoid duplicates
    (e.g., only bin0-bin1, not both bin0-bin1 and bin1-bin0).

    :param inferred_galaxy_zdists: List of tomographic redshift bins with their
        associated measurement types.

    :return: List of all valid TwoPointXY combinations.

    :raises ValueError: If duplicate bin names are found in inferred_galaxy_zdists.
    """
    _validate_list_of_inferred_galaxy_zdists(inferred_galaxy_zdists)
    expanded = [
        (igz, m) for igz in inferred_galaxy_zdists for m in igz.measurement_list
    ]

    # Create all combinations of the expanded list, keeping only compatible ones
    # and avoiding duplicates in the case of correlations of the same type
    all_xy = [
        mdt.TwoPointXY(x=igz1, y=igz2, x_measurement=m1, y_measurement=m2)
        for (igz1, m1), (igz2, m2) in product(expanded, repeat=2)
        if mdt.measurement_is_compatible(m1, m2)
        and ((m1 != m2) or (igz2.bin_name >= igz1.bin_name))
    ]

    # Reorder expanded to have alphabetical order considering first measurements, then
    # bin names.
    return sorted(
        all_xy,
        key=lambda xy: (
            xy.x_measurement,
            xy.y_measurement,
            xy.x.bin_name,
            xy.y.bin_name,
        ),
    )


def make_all_photoz_bin_combinations_with_cmb(
    inferred_galaxy_zdists: list[mdt.TomographicBin],
    cmb_tracer_name: str = "cmb_convergence",
    include_cmb_auto: bool = False,
) -> list[mdt.TwoPointXY]:
    """Create all galaxy-galaxy and CMB-galaxy correlation combinations.

    This function generates all possible two-point correlations including both
    galaxy-galaxy auto/cross-correlations and CMB-galaxy cross-correlations.

    :param inferred_galaxy_zdists: List of galaxy redshift bins with their associated
        measurement types.
    :param cmb_tracer_name: Name to assign to the CMB tracer (default:
        "cmb_convergence").
    :param include_cmb_auto: Whether to include CMB auto-correlation (default: False).

    :return: Combined list of galaxy-galaxy and CMB-galaxy correlation combinations.

    :raises ValueError: If duplicate bin names are found in inferred_galaxy_zdists.
    """
    _validate_list_of_inferred_galaxy_zdists(inferred_galaxy_zdists)
    # Get all galaxy-galaxy combinations first
    galaxy_combinations = make_all_photoz_bin_combinations(inferred_galaxy_zdists)
    all_combinations = galaxy_combinations + make_cmb_galaxy_combinations_only(
        inferred_galaxy_zdists, cmb_tracer_name, include_cmb_auto
    )

    return all_combinations


def make_cmb_galaxy_combinations_only(
    inferred_galaxy_zdists: list[mdt.TomographicBin],
    cmb_tracer_name: str = "cmb_convergence",
    include_cmb_auto: bool = False,
) -> list[mdt.TwoPointXY]:
    """Create only CMB-galaxy cross-correlations.

    This function generates cross-correlations between CMB convergence and galaxy
    measurements, optionally including the CMB auto-correlation. It does NOT include
    any galaxy-galaxy correlations.

    :param inferred_galaxy_zdists: List of galaxy redshift bins with their
        associated measurement types.
    :param cmb_tracer_name: Name to assign to the CMB tracer (default:
        "cmb_convergence").
    :param include_cmb_auto: Whether to include CMB auto-correlation (default: False).

    :return: List of CMB-galaxy cross-correlation combinations (and optionally CMB
        auto).

    :raises ValueError: If duplicate bin names are found in inferred_galaxy_zdists.
    """
    _validate_list_of_inferred_galaxy_zdists(inferred_galaxy_zdists)
    # Create a mock mdt.CMB "bin"
    cmb_bin = mdt.CMBLensing(
        bin_name=cmb_tracer_name,
        z_lss=1100.0,
        measurements={mdt.CMB.CONVERGENCE},
        type_source=mdt.TypeSource.DEFAULT,
    )

    cmb_galaxy_combinations = []

    x: mdt.ProjectedField
    y: mdt.ProjectedField
    for galaxy_bin in inferred_galaxy_zdists:
        galaxy_bin_type: list[tuple[mdt.ProjectedField, mdt.Measurement]] = list(
            product([galaxy_bin], list(galaxy_bin.measurements))
        )
        cmb_bin_type: list[tuple[mdt.ProjectedField, mdt.Measurement]] = list(
            product([cmb_bin], list(cmb_bin.measurements))
        )
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


def make_binned_two_point_filtered(
    inferred_galaxy_zdists: list[mdt.InferredGalaxyZDist],
    bin_pair_selector: mdt.BinPairSelector,
) -> list[mdt.TwoPointXY]:
    """Create two-point correlations filtered by a bin pair selector.

    This function generates all possible bin combinations and then filters them using
    the provided selector, keeping only pairs that satisfy the selection criteria
    (e.g., auto-correlations only, specific measurements, neighboring bins).

    :param inferred_galaxy_zdists: List of tomographic redshift bins with their
        associated measurement types.
    :param bin_pair_selector: Selector defining which bin pairs to include.

    :return: List of TwoPointXY combinations that pass the selector's criteria.

    :raises ValueError: If duplicate bin names are found in inferred_galaxy_zdists.

    Example:
        # Get only auto-correlations of source measurements
        selector = AutoNameBinPairSelector() & SourceBinPairSelector()
        combinations = make_binned_two_point_filtered(bins, selector)
    """
    _validate_list_of_inferred_galaxy_zdists(inferred_galaxy_zdists)
    all_bin_combinations = make_all_photoz_bin_combinations(inferred_galaxy_zdists)
    return [
        xy
        for xy in all_bin_combinations
        if bin_pair_selector.keep((xy.x, xy.y), (xy.x_measurement, xy.y_measurement))
    ]
