"""Tests for combination utility functions in firecrown.metadata_functions.

This module contains tests for the functions that generate and filter
two-point correlation combinations from tomographic bins.
"""

import pytest
import numpy as np

import firecrown.metadata_types as mt
from firecrown.metadata_functions import (
    filter_two_point_combinations,
    make_all_photoz_bin_combinations,
)


@pytest.fixture(name="sample_combinations")
def fixture_sample_combinations(
    all_harmonic_bins: list[mt.InferredGalaxyZDist],
) -> list[mt.TwoPointXY]:
    """Create a sample list of TwoPointXY combinations for testing."""
    return make_all_photoz_bin_combinations(all_harmonic_bins)


def test_filter_two_point_combinations_empty_input():
    """Test filtering with an empty input list."""
    selector = mt.AutoNameBinPairSelector()
    result = filter_two_point_combinations([], selector)
    assert result == []


def test_filter_two_point_combinations_all_pass(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering where all combinations pass the selector."""
    # Create a selector that accepts everything
    # Using OR of complementary selectors
    selector = mt.AutoBinPairSelector() | mt.CrossBinPairSelector()

    result = filter_two_point_combinations(sample_combinations, selector)

    # All combinations should pass
    assert len(result) == len(sample_combinations)
    assert result == sample_combinations


def test_filter_two_point_combinations_none_pass(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering where no combinations pass the selector."""
    # Create a selector that rejects everything by using contradictory conditions
    selector = mt.AutoNameBinPairSelector() & mt.CrossNameBinPairSelector()

    result = filter_two_point_combinations(sample_combinations, selector)

    # No combinations should pass (can't be both auto and cross in name)
    assert len(result) == 0
    assert result == []


def test_filter_two_point_combinations_auto_name(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering with AutoNameBinPairSelector."""
    selector = mt.AutoNameBinPairSelector()

    result = filter_two_point_combinations(sample_combinations, selector)

    # Verify all results are auto-correlations (same bin names)
    assert len(result) > 0
    for combo in result:
        assert combo.x.bin_name == combo.y.bin_name

    # Verify completeness - all auto-name combinations should be present
    auto_from_sample = [c for c in sample_combinations if c.x.bin_name == c.y.bin_name]
    assert len(result) == len(auto_from_sample)


def test_filter_two_point_combinations_cross_name(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering with CrossNameBinPairSelector."""
    selector = mt.CrossNameBinPairSelector()

    result = filter_two_point_combinations(sample_combinations, selector)

    # Verify all results are cross-correlations (different bin names)
    assert len(result) > 0
    for combo in result:
        assert combo.x.bin_name != combo.y.bin_name

    # Verify completeness - all cross-name combinations should be present
    cross_from_sample = [c for c in sample_combinations if c.x.bin_name != c.y.bin_name]
    assert len(result) == len(cross_from_sample)


def test_filter_two_point_combinations_source_selector(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering with SourceBinPairSelector."""
    selector = mt.SourceBinPairSelector()

    result = filter_two_point_combinations(sample_combinations, selector)

    # Verify all results have source measurements
    assert len(result) > 0
    for combo in result:
        assert combo.x_measurement in mt.GALAXY_SOURCE_TYPES
        assert combo.y_measurement in mt.GALAXY_SOURCE_TYPES


def test_filter_two_point_combinations_lens_selector(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering with LensBinPairSelector."""
    selector = mt.LensBinPairSelector()

    result = filter_two_point_combinations(sample_combinations, selector)

    # Verify all results have lens measurements
    assert len(result) > 0
    for combo in result:
        assert combo.x_measurement in mt.GALAXY_LENS_TYPES
        assert combo.y_measurement in mt.GALAXY_LENS_TYPES


def test_filter_two_point_combinations_composite_and(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering with composite AND selector."""
    selector = mt.AutoNameBinPairSelector() & mt.SourceBinPairSelector()

    result = filter_two_point_combinations(sample_combinations, selector)

    # Verify all results satisfy both conditions
    assert len(result) > 0
    for combo in result:
        # Auto-correlation (same bin name)
        assert combo.x.bin_name == combo.y.bin_name
        # Source measurements
        assert combo.x_measurement in mt.GALAXY_SOURCE_TYPES
        assert combo.y_measurement in mt.GALAXY_SOURCE_TYPES


def test_filter_two_point_combinations_composite_or(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering with composite OR selector."""
    selector = (mt.AutoNameBinPairSelector() & mt.SourceBinPairSelector()) | (
        mt.AutoNameBinPairSelector() & mt.LensBinPairSelector()
    )

    result = filter_two_point_combinations(sample_combinations, selector)

    # Verify all results satisfy at least one condition
    assert len(result) > 0
    for combo in result:
        # All should be auto-correlations
        assert combo.x.bin_name == combo.y.bin_name
        # Should be either source or lens
        is_source = (
            combo.x_measurement in mt.GALAXY_SOURCE_TYPES
            and combo.y_measurement in mt.GALAXY_SOURCE_TYPES
        )
        is_lens = (
            combo.x_measurement in mt.GALAXY_LENS_TYPES
            and combo.y_measurement in mt.GALAXY_LENS_TYPES
        )
        assert is_source or is_lens


def test_filter_two_point_combinations_not_selector(
    sample_combinations: list[mt.TwoPointXY],
):
    """Test filtering with NOT selector."""
    selector = ~mt.AutoNameBinPairSelector()

    result = filter_two_point_combinations(sample_combinations, selector)

    # Verify none of the results are auto-correlations
    for combo in result:
        assert combo.x.bin_name != combo.y.bin_name


def test_filter_two_point_combinations_named_selector():
    """Test filtering with NamedBinPairSelector."""
    # Create specific bins for this test
    bin1 = mt.InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    bin2 = mt.InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    bin3 = mt.InferredGalaxyZDist(
        bin_name="bin_3",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )

    combinations = make_all_photoz_bin_combinations([bin1, bin2, bin3])

    # Select only bin_1 with bin_2
    selector = mt.NamedBinPairSelector(names=[("bin_1", "bin_2")])
    result = filter_two_point_combinations(combinations, selector)

    # Should only have the bin_1-bin_2 pair (note: order matters for
    # NamedBinPairSelector)
    assert len(result) == 1
    assert result[0].x.bin_name == "bin_1"
    assert result[0].y.bin_name == "bin_2"


def test_filter_two_point_combinations_preserves_order():
    """Test that filtering preserves the original order of combinations."""
    # Create bins with specific names to control ordering
    bins = [
        mt.InferredGalaxyZDist(
            bin_name=f"bin_{i}",
            z=np.array([0.1, 0.2, 0.3]),
            dndz=np.array([1.0, 2.0, 1.0]),
            measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
        )
        for i in range(3)
    ]

    combinations = make_all_photoz_bin_combinations(bins)

    # Filter to get a subset
    selector = mt.SourceBinPairSelector()
    result = filter_two_point_combinations(combinations, selector)

    # Extract the source combinations from the original list
    expected_order = [
        c
        for c in combinations
        if c.x_measurement in mt.GALAXY_SOURCE_TYPES
        and c.y_measurement in mt.GALAXY_SOURCE_TYPES
    ]

    # Verify order is preserved
    assert result == expected_order


def test_filter_two_point_combinations_auto_measurement():
    """Test filtering with AutoMeasurementBinPairSelector."""
    bin1 = mt.InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
    )
    bin2 = mt.InferredGalaxyZDist(
        bin_name="bin_2",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
    )

    combinations = make_all_photoz_bin_combinations([bin1, bin2])

    selector = mt.AutoMeasurementBinPairSelector()
    result = filter_two_point_combinations(combinations, selector)

    # Verify all results have the same measurement type
    for combo in result:
        assert combo.x_measurement == combo.y_measurement


def test_filter_two_point_combinations_cross_measurement():
    """Test filtering with CrossMeasurementBinPairSelector."""
    bin1 = mt.InferredGalaxyZDist(
        bin_name="bin_1",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
    )

    combinations = make_all_photoz_bin_combinations([bin1])

    selector = mt.CrossMeasurementBinPairSelector()
    result = filter_two_point_combinations(combinations, selector)

    # Verify all results have different measurement types
    for combo in result:
        assert combo.x_measurement != combo.y_measurement


def test_filter_two_point_combinations_complex_selector():
    """Test filtering with a complex nested selector."""
    bins = [
        mt.InferredGalaxyZDist(
            bin_name=f"bin_{i}",
            z=np.array([0.1, 0.2, 0.3]),
            dndz=np.array([1.0, 2.0, 1.0]),
            measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
        )
        for i in range(3)
    ]

    combinations = make_all_photoz_bin_combinations(bins)

    # Complex selector: (auto-name AND source) OR (cross-name AND lens)
    selector = (mt.AutoNameBinPairSelector() & mt.SourceBinPairSelector()) | (
        mt.CrossNameBinPairSelector() & mt.LensBinPairSelector()
    )

    result = filter_two_point_combinations(combinations, selector)

    assert len(result) > 0
    for combo in result:
        # Check if it satisfies one of the two conditions
        auto_source = (
            combo.x.bin_name == combo.y.bin_name
            and combo.x_measurement in mt.GALAXY_SOURCE_TYPES
            and combo.y_measurement in mt.GALAXY_SOURCE_TYPES
        )
        cross_lens = (
            combo.x.bin_name != combo.y.bin_name
            and combo.x_measurement in mt.GALAXY_LENS_TYPES
            and combo.y_measurement in mt.GALAXY_LENS_TYPES
        )
        assert auto_source or cross_lens


def test_filter_two_point_combinations_triple_negation():
    """Test filtering with triple negation selector."""
    bins = [
        mt.InferredGalaxyZDist(
            bin_name=f"bin_{i}",
            z=np.array([0.1, 0.2, 0.3]),
            dndz=np.array([1.0, 2.0, 1.0]),
            measurements={mt.Galaxies.SHEAR_E},
        )
        for i in range(2)
    ]

    combinations = make_all_photoz_bin_combinations(bins)

    # Triple negation: ~~~AutoNameBinPairSelector() == ~AutoNameBinPairSelector()
    selector = ~~~mt.AutoNameBinPairSelector()
    result = filter_two_point_combinations(combinations, selector)

    # Should be equivalent to NOT auto-name
    for combo in result:
        assert combo.x.bin_name != combo.y.bin_name


def test_filter_two_point_combinations_with_real_vs_harmonic():
    """Test that filter_two_point_combinations.

    Test that filter_two_point_combinations works with different measurement
    types.
    """
    # Create bins with measurements that support different spaces
    harmonic_bin = mt.InferredGalaxyZDist(
        bin_name="harmonic",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
    )

    combinations = make_all_photoz_bin_combinations([harmonic_bin])

    # Filter by source measurements
    selector = mt.SourceBinPairSelector()
    result = filter_two_point_combinations(combinations, selector)

    # Should only include shear-shear combinations
    assert len(result) == 1
    assert result[0].x_measurement == mt.Galaxies.SHEAR_E
    assert result[0].y_measurement == mt.Galaxies.SHEAR_E


def test_filter_two_point_combinations_left_right_measurement_selectors():
    """Test filtering with SourceLensBinPairSelector."""
    bin1 = mt.InferredGalaxyZDist(
        bin_name="src",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.SHEAR_E},
    )
    bin2 = mt.InferredGalaxyZDist(
        bin_name="lens",
        z=np.array([0.1, 0.2, 0.3]),
        dndz=np.array([1.0, 2.0, 1.0]),
        measurements={mt.Galaxies.COUNTS},
    )

    combinations = make_all_photoz_bin_combinations([bin1, bin2])

    # SourceLensBinPairSelector combines source on left and lens on right
    selector = mt.SourceLensBinPairSelector()

    result = filter_two_point_combinations(combinations, selector)

    # Should have exactly one pair: (source, lens)
    assert len(result) == 1
    assert result[0].x.bin_name == "src"
    assert result[0].y.bin_name == "lens"
    assert result[0].x_measurement in mt.GALAXY_SOURCE_TYPES
    assert result[0].y_measurement in mt.GALAXY_LENS_TYPES


def test_filter_two_point_combinations_idempotent():
    """Test that applying the same filter twice gives the same result."""
    bins = [
        mt.InferredGalaxyZDist(
            bin_name=f"bin_{i}",
            z=np.array([0.1, 0.2, 0.3]),
            dndz=np.array([1.0, 2.0, 1.0]),
            measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
        )
        for i in range(3)
    ]

    combinations = make_all_photoz_bin_combinations(bins)

    selector = mt.SourceBinPairSelector()
    result1 = filter_two_point_combinations(combinations, selector)
    result2 = filter_two_point_combinations(result1, selector)

    # Applying the filter twice should give the same result
    assert result1 == result2


def test_filter_two_point_combinations_commutative_and():
    """Test that AND selector is commutative."""
    bins = [
        mt.InferredGalaxyZDist(
            bin_name=f"bin_{i}",
            z=np.array([0.1, 0.2, 0.3]),
            dndz=np.array([1.0, 2.0, 1.0]),
            measurements={mt.Galaxies.SHEAR_E},
        )
        for i in range(2)
    ]

    combinations = make_all_photoz_bin_combinations(bins)

    selector1 = mt.AutoNameBinPairSelector() & mt.SourceBinPairSelector()
    selector2 = mt.SourceBinPairSelector() & mt.AutoNameBinPairSelector()

    result1 = filter_two_point_combinations(combinations, selector1)
    result2 = filter_two_point_combinations(combinations, selector2)

    assert result1 == result2


def test_filter_two_point_combinations_commutative_or():
    """Test that OR selector is commutative."""
    bins = [
        mt.InferredGalaxyZDist(
            bin_name=f"bin_{i}",
            z=np.array([0.1, 0.2, 0.3]),
            dndz=np.array([1.0, 2.0, 1.0]),
            measurements={mt.Galaxies.SHEAR_E, mt.Galaxies.COUNTS},
        )
        for i in range(2)
    ]

    combinations = make_all_photoz_bin_combinations(bins)

    selector1 = mt.SourceBinPairSelector() | mt.LensBinPairSelector()
    selector2 = mt.LensBinPairSelector() | mt.SourceBinPairSelector()

    result1 = filter_two_point_combinations(combinations, selector1)
    result2 = filter_two_point_combinations(combinations, selector2)

    # Results should be identical including order (since
    # make_all_photoz_bin_combinations returns sorted results and filter preserves
    # order)
    assert len(result1) == len(result2)
    assert result1 == result2
