"""Unit tests for cosmological parameter conflict detection."""

import pytest

from firecrown.connector.cosmosis.likelihood import _check_cosmology_conflicts
from collections.abc import Mapping
from typing import cast


def test_no_conflicts_different_keys():
    """Test when merged and canonical have completely different keys."""
    merged = {"omega_c": 0.25, "omega_b": 0.05}
    canonical = {"h": 0.67, "sigma8": 0.81}

    # Should complete without exception
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_no_conflicts_empty_canonical():
    """Test when canonical is empty."""
    merged = {"omega_c": 0.25, "omega_b": 0.05}
    canonical: dict[str, float | list[float]] = {}

    # Should complete without exception
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_no_conflicts_empty_merged():
    """Test when merged is empty."""
    merged: dict[str, float | list[float]] = {}
    canonical = {"Omega_c": 0.25, "Omega_b": 0.05}

    # Should complete without exception
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_scalar_conflict_same_case():
    """Test scalar conflict when keys have the same case."""
    merged = {"Omega_c": 0.25}
    canonical = {"Omega_c": 0.26}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_scalar_conflict_different_case():
    """Test scalar conflict when keys differ in case (case-insensitive match)."""
    merged = {"omega_c": 0.25}
    canonical = {"Omega_c": 0.26}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_scalar_no_conflict_equal_values():
    """Test when scalar values are equal (case-insensitive key match)."""
    merged = {"omega_c": 0.25}
    canonical = {"Omega_c": 0.25}

    # Should complete without exception
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_scalar_no_conflict_equal_values_float_conversion():
    """Test when scalar values are equal after float conversion."""
    merged = {"omega_c": 0.25}
    canonical = {"Omega_c": 0.250000}

    # Should complete without exception
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_list_no_conflict_equal_values():
    """Test when list values are equal (case-insensitive key match)."""
    merged = {"m_nu": [0.06, 0.0, 0.0]}
    canonical = {"m_nu": [0.06, 0.0, 0.0]}

    # Should complete without exception
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_list_no_conflict_equal_after_conversion():
    """Test when list values are equal after float conversion."""
    merged = {"m_nu": [0.06, 0, 0]}
    canonical = {"m_nu": [0.06, 0.0, 0.0]}

    # Should complete without exception
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_list_conflict_different_values():
    """Test list conflict when values differ."""
    merged = {"m_nu": [0.06, 0.0, 0.0]}
    canonical = {"m_nu": [0.07, 0.0, 0.0]}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_list_conflict_different_lengths():
    """Test list conflict when lists have different lengths."""
    merged = {"m_nu": [0.06, 0.0]}
    canonical = {"m_nu": [0.06, 0.0, 0.0]}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_list_conversion_failure_raises():
    """Test when list contains non-numeric values that fail float conversion."""
    merged = {"m_nu": [0.06, "invalid", 0.0]}
    canonical = {"m_nu": [0.06, 0.0, 0.0]}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_list_conversion_failure_in_canonical():
    """Test when canonical list contains non-numeric values."""
    merged = {"m_nu": [0.06, 0.0, 0.0]}
    canonical = {"m_nu": [0.06, "bad", 0.0]}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_mixed_types_scalar_vs_list():
    """Test when one parameter is scalar and the other is a list."""
    merged = {"omega_c": 0.25}
    canonical = {"Omega_c": [0.25]}

    # Should go through scalar comparison path and raise due to type mismatch
    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_mixed_types_list_vs_scalar():
    """Test when one parameter is list and the other is scalar."""
    merged = {"m_nu": [0.06]}
    canonical = {"m_nu": 0.06}

    # Should go through scalar comparison path
    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_case_insensitive_matching():
    """Test various case combinations for key matching."""
    merged = {"omega_c": 0.25, "OMEGA_B": 0.05, "H": 0.67}
    canonical = {"Omega_C": 0.25, "omega_b": 0.05, "h": 0.67}

    # All should match case-insensitively and have equal values
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_multiple_conflicts_first_detected():
    """Test that the first conflict is detected and raised."""
    merged = {"omega_c": 0.25, "omega_b": 0.05}
    canonical = {"Omega_c": 0.26, "Omega_b": 0.06}

    # Should raise on first conflict encountered
    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_partial_overlap_with_conflict():
    """Test when some keys overlap with conflict and others don't."""
    merged = {"omega_c": 0.25, "omega_b": 0.05}
    canonical = {"Omega_c": 0.26, "h": 0.67}

    # Should raise on the conflicting key
    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_partial_overlap_without_conflict():
    """Test when some keys overlap without conflict and others don't."""
    merged = {"omega_c": 0.25, "omega_b": 0.05}
    canonical = {"Omega_c": 0.25, "h": 0.67}

    # Should complete without exception
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_tuple_vs_list_equal():
    """Test when one is tuple and the other is list with equal values."""
    merged = {"m_nu": (0.06, 0.0, 0.0)}
    canonical = {"m_nu": [0.06, 0.0, 0.0]}

    # Should complete without exception (both treated as sequences)
    _check_cosmology_conflicts(
        cast(Mapping[str, float | list[float]], merged),
        cast(Mapping[str, float | list[float]], canonical),
    )


def test_tuple_vs_list_different():
    """Test when one is tuple and the other is list with different values."""
    merged = {"m_nu": (0.06, 0.0, 0.0)}
    canonical = {"m_nu": [0.07, 0.0, 0.0]}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_scalar_conversion_failure():
    """Test when scalar value cannot be converted to float."""
    merged = {"omega_c": "not-a-number"}
    canonical = {"Omega_c": 0.25}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )


def test_canonical_scalar_conversion_failure():
    """Test when canonical scalar value cannot be converted to float."""
    merged = {"omega_c": 0.25}
    canonical = {"Omega_c": "bad-value"}

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        _check_cosmology_conflicts(
            cast(Mapping[str, float | list[float]], merged),
            cast(Mapping[str, float | list[float]], canonical),
        )
