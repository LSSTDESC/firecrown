"""Tests for the firecrown.data_functions module."""

import pytest

from firecrown.data_functions import ensure_no_overlaps


@pytest.mark.parametrize(
    "measurement, index_set, index_sets, other_measurements, expected_result",
    [
        # Case where there's no overlap
        ("A", {1, 2}, [{3, 4}, {5, 6}], ["B", "C"], None),
        # Case where there's an overlap with one set
        ("A", {1, 2}, [{2, 3}, {5, 6}], ["B", "C"], ValueError),
        # Case where there's an overlap with multiple sets
        ("A", {1, 2}, [{1, 3}, {2, 4}], ["B", "C"], ValueError),
        # Case with empty sets to test edge cases
        ("A", set(), [set(), set()], ["B", "C"], None),
        # Case with overlap for the first set only
        ("A", {1}, [{1, 2}, {3, 4}], ["B", "C"], ValueError),
        # Case with overlap for the second set only
        ("A", {3}, [{1, 2}, {3, 4}], ["B", "C"], ValueError),
    ],
)
def test_ensure_no_overlaps(
    measurement, index_set, index_sets, other_measurements, expected_result
):
    if expected_result is None:
        ensure_no_overlaps(measurement, index_set, index_sets, other_measurements)
    else:
        with pytest.raises(expected_result) as excinfo:
            ensure_no_overlaps(measurement, index_set, index_sets, other_measurements)
        assert "overlap" in str(excinfo.value), "Error message should contain 'overlap'"


# Additional test for empty other_measurements and index_sets
def test_empty_other_measurements_and_index_sets():
    ensure_no_overlaps("A", {1, 2}, [], [])


# Additional test for overlap with the last set in the list
def test_overlap_with_last_set():
    with pytest.raises(ValueError) as excinfo:
        ensure_no_overlaps("A", {1}, [{3, 4}, {5, 6}, {1, 7}], ["B", "C", "D"])
    assert "overlap" in str(excinfo.value), "Error message should contain 'overlap'"


# Test for edge case where the index_set is empty but there are non-empty set
# in index_sets
def test_empty_index_set_with_non_empty_index_sets():
    ensure_no_overlaps("A", set(), [{1}, {2}], ["B", "C"])


# Test for edge case where all sets are empty
def test_all_sets_empty():
    ensure_no_overlaps("A", set(), [set(), set()], ["B", "C"])
