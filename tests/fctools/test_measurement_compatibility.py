"""Unit tests for firecrown.fctools.measurement_compatibility module.

Tests the measurement compatibility analysis tool.
"""

import subprocess
import sys

from rich.console import Console

from firecrown.fctools.measurement_compatibility import (
    discover_measurements_by_space,
    generate_compatible_pairs,
    print_compatible_pairs,
    print_efficiency_gains,
    print_measurements_by_space,
    print_summary_stats,
)
from firecrown.metadata_types import (
    ALL_MEASUREMENTS,
    Measurement,
    measurement_is_compatible_harmonic,
    measurement_is_compatible_real,
    measurement_supports_harmonic,
    measurement_supports_real,
)


class TestDiscoverMeasurementsBySpace:
    """Tests for discover_measurements_by_space function."""

    def test_discovers_measurements(self):
        """Test that measurements are discovered correctly."""
        real_measurements, harmonic_measurements = discover_measurements_by_space()

        # Should return two lists
        assert isinstance(real_measurements, list)
        assert isinstance(harmonic_measurements, list)

        # Both lists should contain Measurement objects
        assert all(isinstance(m, Measurement) for m in real_measurements)
        assert all(isinstance(m, Measurement) for m in harmonic_measurements)

    def test_real_measurements_support_real_space(self):
        """Test that all real measurements support real space."""
        real_measurements, _ = discover_measurements_by_space()

        # All real measurements should support real space
        for m in real_measurements:
            assert measurement_supports_real(m), f"{m} should support real space"

    def test_harmonic_measurements_support_harmonic_space(self):
        """Test that all harmonic measurements support harmonic space."""
        _, harmonic_measurements = discover_measurements_by_space()

        # All harmonic measurements should support harmonic space
        for m in harmonic_measurements:
            assert measurement_supports_harmonic(
                m
            ), f"{m} should support harmonic space"

    def test_measurements_are_subset_of_all(self):
        """Test that discovered measurements are subsets of ALL_MEASUREMENTS."""
        real_measurements, harmonic_measurements = discover_measurements_by_space()

        # Both should be subsets of ALL_MEASUREMENTS
        assert set(real_measurements).issubset(set(ALL_MEASUREMENTS))
        assert set(harmonic_measurements).issubset(set(ALL_MEASUREMENTS))

    def test_discovers_nonempty_lists(self):
        """Test that at least some measurements are discovered."""
        real_measurements, harmonic_measurements = discover_measurements_by_space()

        # Firecrown should have at least some measurements in each space
        assert len(real_measurements) > 0, "Should have real-space measurements"
        assert len(harmonic_measurements) > 0, "Should have harmonic-space measurements"


class TestGenerateCompatiblePairs:
    """Tests for generate_compatible_pairs function."""

    def test_generates_pairs_for_real_space(self):
        """Test generating compatible pairs for real space."""
        real_measurements, _ = discover_measurements_by_space()

        pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )

        # Should return a list of tuples
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) for p in pairs)
        assert all(len(p) == 2 for p in pairs)

    def test_generates_pairs_for_harmonic_space(self):
        """Test generating compatible pairs for harmonic space."""
        _, harmonic_measurements = discover_measurements_by_space()

        pairs = generate_compatible_pairs(
            harmonic_measurements, measurement_is_compatible_harmonic
        )

        # Should return a list of tuples
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) for p in pairs)
        assert all(len(p) == 2 for p in pairs)

    def test_all_pairs_are_compatible(self):
        """Test that all generated pairs are actually compatible."""
        real_measurements, _ = discover_measurements_by_space()

        pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )

        # Verify each pair is compatible
        for m1, m2 in pairs:
            assert measurement_is_compatible_real(
                m1, m2
            ), f"{m1} and {m2} should be compatible"

    def test_pairs_are_from_input_measurements(self):
        """Test that pairs contain only measurements from input list."""
        real_measurements, _ = discover_measurements_by_space()

        pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )

        # All measurements in pairs should be from the input list
        for m1, m2 in pairs:
            assert m1 in real_measurements
            assert m2 in real_measurements

    def test_empty_input_returns_empty_pairs(self):
        """Test that empty measurement list returns empty pairs."""
        pairs = generate_compatible_pairs([], measurement_is_compatible_real)

        assert pairs == []

    def test_single_measurement_returns_compatible_pairs(self):
        """Test behavior with a single measurement."""
        # Get one measurement
        real_measurements, _ = discover_measurements_by_space()
        if real_measurements:
            single = [real_measurements[0]]

            pairs = generate_compatible_pairs(single, measurement_is_compatible_real)

            # Should test the measurement with itself
            if measurement_is_compatible_real(single[0], single[0]):
                assert len(pairs) == 1
                assert pairs[0] == (single[0], single[0])
            else:
                assert len(pairs) == 0


class TestPrintMeasurementsBySpace:
    """Tests for print_measurements_by_space function."""

    def test_prints_counts_non_verbose(self, capsys):
        """Test printing measurement counts in non-verbose mode."""
        console = Console()
        real_measurements, harmonic_measurements = discover_measurements_by_space()

        print_measurements_by_space(
            console, real_measurements, harmonic_measurements, False
        )

        captured = capsys.readouterr()
        assert "Real-space measurements found:" in captured.out
        assert "Harmonic-space measurements found:" in captured.out
        assert str(len(real_measurements)) in captured.out
        assert str(len(harmonic_measurements)) in captured.out

    def test_prints_details_verbose(self, capsys):
        """Test printing measurement details in verbose mode."""
        console = Console()
        real_measurements, harmonic_measurements = discover_measurements_by_space()

        print_measurements_by_space(
            console, real_measurements, harmonic_measurements, True
        )

        captured = capsys.readouterr()
        assert "Real-space measurements found:" in captured.out
        assert "Harmonic-space measurements found:" in captured.out

        # In verbose mode, should show individual measurements
        if real_measurements:
            # Should have bullet points
            assert "•" in captured.out


class TestPrintCompatiblePairs:
    """Tests for print_compatible_pairs function."""

    def test_prints_pair_count_non_verbose(self, capsys):
        """Test printing pair counts in non-verbose mode."""
        console = Console()
        real_measurements, _ = discover_measurements_by_space()
        pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )

        print_compatible_pairs(console, "real-space", pairs, False)

        captured = capsys.readouterr()
        assert "Valid real-space pairs:" in captured.out
        assert str(len(pairs)) in captured.out

    def test_prints_pair_details_verbose(self, capsys):
        """Test printing pair details in verbose mode."""
        console = Console()
        real_measurements, _ = discover_measurements_by_space()
        pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )

        print_compatible_pairs(console, "real-space", pairs, True)

        captured = capsys.readouterr()
        assert "Valid real-space pairs:" in captured.out

        # In verbose mode, should show individual pairs
        if pairs:
            assert "•" in captured.out
            # Should show measurement names
            assert "+" in captured.out

    def test_handles_empty_pairs(self, capsys):
        """Test handling empty pairs list."""
        console = Console()
        print_compatible_pairs(console, "test-space", [], False)

        captured = capsys.readouterr()
        assert "Valid test-space pairs: 0" in captured.out


class TestPrintEfficiencyGains:
    """Tests for print_efficiency_gains function."""

    def test_prints_efficiency_stats(self, capsys):
        """Test printing efficiency statistics."""
        console = Console()
        real_measurements, harmonic_measurements = discover_measurements_by_space()
        real_pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )
        harmonic_pairs = generate_compatible_pairs(
            harmonic_measurements, measurement_is_compatible_harmonic
        )

        print_efficiency_gains(console, real_measurements, real_pairs, harmonic_pairs)

        captured = capsys.readouterr()
        assert "Efficiency Improvements:" in captured.out
        assert "Real space:" in captured.out
        assert "Harmonic space:" in captured.out
        assert "Total:" in captured.out
        assert "skipped tests eliminated" in captured.out

    def test_calculates_correct_reduction(self, capsys):
        """Test that reduction calculations are correct."""
        console = Console()
        real_measurements, harmonic_measurements = discover_measurements_by_space()
        real_pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )
        harmonic_pairs = generate_compatible_pairs(
            harmonic_measurements, measurement_is_compatible_harmonic
        )

        print_efficiency_gains(console, real_measurements, real_pairs, harmonic_pairs)

        captured = capsys.readouterr()

        # Calculate expected reductions
        real_reduction = len(real_measurements) ** 2 - len(real_pairs)
        harmonic_reduction = len(harmonic_measurements) ** 2 - len(harmonic_pairs)

        assert str(real_reduction) in captured.out
        assert str(harmonic_reduction) in captured.out
        assert str(real_reduction + harmonic_reduction) in captured.out


class TestPrintSummaryStats:
    """Tests for print_summary_stats function."""

    def test_prints_summary_statistics(self, capsys):
        """Test printing summary statistics."""
        console = Console()
        real_measurements, harmonic_measurements = discover_measurements_by_space()
        real_pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )
        harmonic_pairs = generate_compatible_pairs(
            harmonic_measurements, measurement_is_compatible_harmonic
        )

        print_summary_stats(
            console,
            real_measurements,
            harmonic_measurements,
            real_pairs,
            harmonic_pairs,
        )

        captured = capsys.readouterr()
        assert "Summary Statistics:" in captured.out
        assert "Total measurements:" in captured.out
        assert "Real-space coverage:" in captured.out
        assert "Harmonic-space coverage:" in captured.out
        assert "Real-space compatibility:" in captured.out
        assert "Harmonic-space compatibility:" in captured.out

    def test_shows_percentages(self, capsys):
        """Test that percentages are shown."""
        console = Console()
        real_measurements, harmonic_measurements = discover_measurements_by_space()
        real_pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )
        harmonic_pairs = generate_compatible_pairs(
            harmonic_measurements, measurement_is_compatible_harmonic
        )

        print_summary_stats(
            console,
            real_measurements,
            harmonic_measurements,
            real_pairs,
            harmonic_pairs,
        )

        captured = capsys.readouterr()
        # Should contain percentage symbols
        assert "%" in captured.out


class TestMainFunction:  # pylint: disable=import-outside-toplevel
    """Tests for main CLI function."""

    def test_main_default_options(self, capsys):
        """Test main with default options (space=BOTH, verbose=False)."""
        from firecrown.fctools.measurement_compatibility import Space, main

        # Call main directly with default options
        main(verbose=False, space=Space.BOTH, stats_only=False)

        captured = capsys.readouterr()
        assert "Firecrown Measurement Compatibility Analysis" in captured.out
        assert "Real-space measurements found:" in captured.out
        assert "Harmonic-space measurements found:" in captured.out
        assert "Efficiency Improvements:" in captured.out
        assert "Summary Statistics:" in captured.out

    def test_main_verbose_flag(self, capsys):
        """Test main with verbose=True."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=True, space=Space.BOTH, stats_only=False)

        captured = capsys.readouterr()
        assert "Firecrown Measurement Compatibility Analysis" in captured.out
        # In verbose mode, should show individual measurements
        assert "•" in captured.out

    def test_main_verbose_with_space_harmonic(self, capsys):
        """Test main with verbose=True and space=HARMONIC."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=True, space=Space.HARMONIC, stats_only=False)

        captured = capsys.readouterr()
        assert "•" in captured.out
        assert "harmonic-space" in captured.out

    def test_main_space_real(self, capsys):
        """Test main with space=REAL."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.REAL, stats_only=False)

        captured = capsys.readouterr()
        assert "real-space" in captured.out
        # Should show Real-space efficiency
        assert "Real-space efficiency:" in captured.out
        # Should not show harmonic-space pairs
        assert "harmonic-space" not in captured.out

    def test_main_space_harmonic(self, capsys):
        """Test main with space=HARMONIC."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.HARMONIC, stats_only=False)

        captured = capsys.readouterr()
        assert "harmonic-space" in captured.out
        # Should show efficiency for harmonic space
        assert "Harmonic-space efficiency:" in captured.out

    def test_main_space_both(self, capsys):
        """Test main with space=BOTH."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.BOTH, stats_only=False)

        captured = capsys.readouterr()
        assert "real-space" in captured.out
        assert "harmonic-space" in captured.out
        assert "Summary Statistics:" in captured.out

    def test_main_stats_only(self, capsys):
        """Test main with stats_only=True."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.BOTH, stats_only=True)

        captured = capsys.readouterr()
        assert "Summary Statistics:" in captured.out
        # Should not show detailed pair listings
        assert "Valid real-space pairs:" not in captured.out
        # Should not show efficiency improvements
        assert "Efficiency Improvements:" not in captured.out

    def test_main_stats_only_with_space_real(self, capsys):
        """Test main with stats_only=True and space=REAL."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.REAL, stats_only=True)

        captured = capsys.readouterr()
        assert "Summary Statistics:" in captured.out
        # Should not show pair listings or efficiency when stats-only
        assert "Valid real-space pairs:" not in captured.out
        assert "Real-space efficiency:" not in captured.out

    def test_main_stats_only_with_space_harmonic(self, capsys):
        """Test main with stats_only=True and space=HARMONIC."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.HARMONIC, stats_only=True)

        captured = capsys.readouterr()
        assert "Summary Statistics:" in captured.out
        # Should not show pair listings or efficiency when stats-only
        assert "Valid harmonic-space pairs:" not in captured.out
        assert "Harmonic-space efficiency:" not in captured.out

    def test_main_combined_flags(self, capsys):
        """Test main with verbose=True and space=REAL."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=True, space=Space.REAL, stats_only=False)

        captured = capsys.readouterr()
        assert "•" in captured.out
        assert "real-space" in captured.out
        # Should show Real-space efficiency
        assert "Real-space efficiency:" in captured.out

    def test_main_space_real_without_verbose(self, capsys):
        """Test main with space=REAL and verbose=False."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.REAL, stats_only=False)

        captured = capsys.readouterr()
        assert "Real-space measurements found:" in captured.out
        assert "Valid real-space pairs:" in captured.out
        # Should show Real-space efficiency
        assert "Real-space efficiency:" in captured.out

    def test_main_space_harmonic_without_verbose(self, capsys):
        """Test main with space=HARMONIC and verbose=False."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.HARMONIC, stats_only=False)

        captured = capsys.readouterr()
        assert "harmonic-space" in captured.out
        assert "Harmonic-space efficiency:" in captured.out
        # Should not show real-space details when space=harmonic
        assert "Valid real-space pairs:" not in captured.out

    def test_main_space_both_shows_all_sections(self, capsys):
        """Test main with space=BOTH shows all sections."""
        from firecrown.fctools.measurement_compatibility import Space, main

        main(verbose=False, space=Space.BOTH, stats_only=False)

        captured = capsys.readouterr()
        # Should show measurements by space
        assert "Real-space measurements found:" in captured.out
        assert "Harmonic-space measurements found:" in captured.out
        # Should show both pair types
        assert "Valid real-space pairs:" in captured.out
        assert "Valid harmonic-space pairs:" in captured.out
        # Should show efficiency gains section
        assert "Efficiency Improvements:" in captured.out
        # Should show summary statistics
        assert "Summary Statistics:" in captured.out

    def test_main_with_subprocess(self):
        """Test that the script can be executed directly via subprocess.

        This test verifies that the __main__ block works correctly.
        """
        script_path = "firecrown/fctools/measurement_compatibility.py"
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Firecrown Measurement Compatibility Analysis" in result.stdout

    def test_main_subprocess_with_stats_only(self):
        """Test script execution with --stats-only flag."""
        script_path = "firecrown/fctools/measurement_compatibility.py"
        result = subprocess.run(
            [sys.executable, script_path, "--stats-only"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Summary Statistics:" in result.stdout


class TestIntegration:
    """Integration tests for measurement compatibility functionality."""

    def test_full_workflow_real_space(self):
        """Test complete workflow for real space."""
        # Discover measurements
        real_measurements, _ = discover_measurements_by_space()

        # Generate compatible pairs
        pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )

        # Verify all pairs are valid
        assert len(pairs) > 0
        assert len(pairs) <= len(real_measurements) ** 2

        # Verify compatibility
        for m1, m2 in pairs:
            assert measurement_is_compatible_real(m1, m2)

    def test_full_workflow_harmonic_space(self):
        """Test complete workflow for harmonic space."""
        # Discover measurements
        _, harmonic_measurements = discover_measurements_by_space()

        # Generate compatible pairs
        pairs = generate_compatible_pairs(
            harmonic_measurements, measurement_is_compatible_harmonic
        )

        # Verify all pairs are valid
        assert len(pairs) > 0
        assert len(pairs) <= len(harmonic_measurements) ** 2

        # Verify compatibility
        for m1, m2 in pairs:
            assert measurement_is_compatible_harmonic(m1, m2)

    def test_compatibility_is_subset_of_all_combinations(self):
        """Test that compatible pairs are a subset of all possible combinations."""
        real_measurements, harmonic_measurements = discover_measurements_by_space()

        real_pairs = generate_compatible_pairs(
            real_measurements, measurement_is_compatible_real
        )
        harmonic_pairs = generate_compatible_pairs(
            harmonic_measurements, measurement_is_compatible_harmonic
        )

        # Compatible pairs should be <= all possible combinations
        assert len(real_pairs) <= len(real_measurements) ** 2
        assert len(harmonic_pairs) <= len(harmonic_measurements) ** 2

    def test_cli_produces_consistent_output(self):
        """Test that CLI produces consistent output across runs."""
        script_path = "firecrown/fctools/measurement_compatibility.py"

        # Run twice
        result1 = subprocess.run(
            [sys.executable, script_path, "--stats-only"],
            capture_output=True,
            text=True,
            check=False,
        )
        result2 = subprocess.run(
            [sys.executable, script_path, "--stats-only"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should produce identical output
        assert result1.returncode == 0
        assert result2.returncode == 0
        assert result1.stdout == result2.stdout

    def test_all_measurements_categorized(self):
        """Test that measurements are properly categorized."""
        real_measurements, harmonic_measurements = discover_measurements_by_space()

        # Count measurements in each category
        real_count = sum(1 for m in ALL_MEASUREMENTS if measurement_supports_real(m))
        harmonic_count = sum(
            1 for m in ALL_MEASUREMENTS if measurement_supports_harmonic(m)
        )

        assert len(real_measurements) == real_count
        assert len(harmonic_measurements) == harmonic_count
