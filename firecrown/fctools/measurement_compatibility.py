#!/usr/bin/env python
"""Tool for analyzing measurement compatibility in Firecrown.

This tool provides insights into which measurement pairs are compatible
for real-space and harmonic-space two-point correlation functions.
Useful for understanding test fixture parameterization and debugging
measurement combination issues.

Physical Examples:
  Compatible: Galaxy counts × CMB convergence (galaxy-CMB lensing)
  Compatible: Galaxy shear × Galaxy counts (galaxy-galaxy lensing)
  Incompatible: ξ₊ × ξ₋ (different correlation function components)
  Incompatible: E-mode shear in real space (harmonic-space only)
"""

from collections.abc import Callable
from enum import Enum
from itertools import product

import typer
from rich.console import Console

from firecrown.metadata_types import (
    ALL_MEASUREMENTS,
    Measurement,
)
from firecrown.metadata_types._compatibility import (
    measurement_is_compatible_harmonic,
    measurement_is_compatible_real,
    _measurement_supports_harmonic,
    _measurement_supports_real,
)


class Space(str, Enum):
    """Enum for space types."""

    REAL = "real"
    HARMONIC = "harmonic"
    BOTH = "both"


def discover_measurements_by_space() -> tuple[list[Measurement], list[Measurement]]:
    """Discover all measurements that support real/harmonic space.

    Returns:
        Tuple of (real_measurements, harmonic_measurements)
    """
    all_measurements = ALL_MEASUREMENTS

    # Categorize by space support
    real_measurements = [m for m in all_measurements if _measurement_supports_real(m)]
    harmonic_measurements = [
        m for m in all_measurements if _measurement_supports_harmonic(m)
    ]

    return real_measurements, harmonic_measurements


def generate_compatible_pairs(
    measurements: list[Measurement],
    compatibility_func: Callable[[Measurement, Measurement], bool],
) -> list[tuple[Measurement, Measurement]]:
    """Generate all valid measurement pairs for a given compatibility function.

    Args:
        measurements: List of measurements to test combinations of
        compatibility_func: Function to test compatibility

    Returns:
        List of valid (measurement1, measurement2) tuples
    """
    return [
        (m1, m2)
        for m1, m2 in product(measurements, repeat=2)
        if compatibility_func(m1, m2)
    ]


def print_measurements_by_space(
    console: Console,
    real_measurements: list[Measurement],
    harmonic_measurements: list[Measurement],
    verbose: bool = False,
) -> None:
    """Print measurements categorized by space support."""
    console.print(f"📊 Real-space measurements found: {len(real_measurements)}")
    if verbose:
        for m in real_measurements:
            console.print(f"   • {m}")

    console.print(
        f"\n📊 Harmonic-space measurements found: {len(harmonic_measurements)}"
    )
    if verbose:
        for m in harmonic_measurements:
            console.print(f"   • {m}")


def print_compatible_pairs(
    console: Console,
    space_name: str,
    pairs: list[tuple[Measurement, Measurement]],
    verbose: bool = False,
) -> None:
    """Print compatible measurement pairs for a given space."""
    console.print(f"\n✅ Valid {space_name} pairs: {len(pairs)}")
    if verbose:
        for m1, m2 in pairs:
            console.print(f"   • {m1.name} + {m2.name}")


def print_efficiency_gains(
    console: Console,
    real_measurements: list[Measurement],
    harmonic_measurements: list[Measurement],
    real_pairs: list[tuple[Measurement, Measurement]],
    harmonic_pairs: list[tuple[Measurement, Measurement]],
) -> None:
    """Print efficiency improvements from using compatible pairs."""
    total_real_combinations = len(real_measurements) ** 2
    total_harmonic_combinations = len(harmonic_measurements) ** 2

    real_skip_reduction = total_real_combinations - len(real_pairs)
    harmonic_skip_reduction = total_harmonic_combinations - len(harmonic_pairs)

    console.print("\n🚀 Efficiency Improvements:")
    console.print(f"   Real space: {real_skip_reduction} skipped tests eliminated")
    console.print(
        f"   Harmonic space: {harmonic_skip_reduction} skipped tests eliminated"
    )
    total_reduction = real_skip_reduction + harmonic_skip_reduction
    console.print(f"   Total: {total_reduction} fewer skipped tests!")


def print_summary_stats(
    console: Console,
    real_measurements: list[Measurement],
    harmonic_measurements: list[Measurement],
    real_pairs: list[tuple[Measurement, Measurement]],
    harmonic_pairs: list[tuple[Measurement, Measurement]],
) -> None:
    """Print summary statistics."""
    total_measurements = len(ALL_MEASUREMENTS)
    real_coverage = len(real_measurements) / total_measurements * 100
    harmonic_coverage = len(harmonic_measurements) / total_measurements * 100

    console.print("\n📈 Summary Statistics:")
    console.print(f"   Total measurements: {total_measurements}")
    console.print(f"   Real-space coverage: {real_coverage:.1f}%")
    console.print(f"   Harmonic-space coverage: {harmonic_coverage:.1f}%")

    # Calculate compatibility rates
    real_total = len(real_measurements) ** 2
    real_rate = len(real_pairs) / real_total * 100
    console.print(
        f"   Real-space compatibility: {len(real_pairs)}/{real_total} "
        f"({real_rate:.1f}%)"
    )

    harmonic_total = len(harmonic_measurements) ** 2
    harmonic_rate = len(harmonic_pairs) / harmonic_total * 100
    console.print(
        f"   Harmonic-space compatibility: {len(harmonic_pairs)}/{harmonic_total} "
        f"({harmonic_rate:.1f}%)"
    )


app = typer.Typer()


@app.command()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed list of measurements and pairs"
    ),
    space: Space = typer.Option(
        Space.BOTH,
        "--space",
        case_sensitive=False,
        help="Which space to analyze",
    ),
    stats_only: bool = typer.Option(
        False, "--stats-only", help="Show only summary statistics"
    ),
) -> None:
    """Analyze measurement compatibility for Firecrown two-point functions.

    This tool discovers all measurement types and analyzes which combinations
    are compatible for real-space and harmonic-space two-point correlation
    functions. It shows efficiency improvements from using pre-filtered
    measurement pairs instead of runtime compatibility checks.

    Compatible correlations include:
      ✅ Galaxy counts × Galaxy counts (galaxy clustering)
      ✅ Galaxy shear × Galaxy shear (cosmic shear)
      ✅ Galaxy counts × Galaxy shear (galaxy-galaxy lensing)
      ✅ CMB convergence × Galaxy counts (galaxy-CMB lensing)
      ✅ Cluster counts × Galaxy counts (cluster-galaxy correlation)

    Incompatible combinations:
      ❌ ξ₊ × ξ₋ (different shear correlation components)
      ❌ SHEAR_T × SHEAR_T (T-mode auto-correlation not measured)
      ❌ E-mode shear in real space (harmonic-space only)

    Examples:
      python -m firecrown.fctools.measurement_compatibility --verbose
      python -m firecrown.fctools.measurement_compatibility --space real
      python -m firecrown.fctools.measurement_compatibility --stats-only
    """
    console = Console()
    console.print("🔍 Firecrown Measurement Compatibility Analysis")
    console.print("=" * 60)

    # Discover measurements
    real_measurements, harmonic_measurements = discover_measurements_by_space()

    # Generate compatible pairs
    real_pairs = generate_compatible_pairs(
        real_measurements, measurement_is_compatible_real
    )
    harmonic_pairs = generate_compatible_pairs(
        harmonic_measurements, measurement_is_compatible_harmonic
    )

    if stats_only:
        print_summary_stats(
            console,
            real_measurements,
            harmonic_measurements,
            real_pairs,
            harmonic_pairs,
        )
        return

    # Print measurements by space
    if space in [Space.REAL, Space.BOTH]:
        print_measurements_by_space(
            console, real_measurements, harmonic_measurements, verbose
        )

    # Print compatible pairs
    if space in [Space.REAL, Space.BOTH]:
        print_compatible_pairs(console, "real-space", real_pairs, verbose)

    if space in [Space.HARMONIC, Space.BOTH]:
        print_compatible_pairs(console, "harmonic-space", harmonic_pairs, verbose)

    # Print efficiency gains
    if space == Space.BOTH:
        print_efficiency_gains(
            console,
            real_measurements,
            harmonic_measurements,
            real_pairs,
            harmonic_pairs,
        )
        print_summary_stats(
            console,
            real_measurements,
            harmonic_measurements,
            real_pairs,
            harmonic_pairs,
        )
    elif space == Space.REAL:
        real_skip_reduction = len(real_measurements) ** 2 - len(real_pairs)
        console.print(
            f"\n🚀 Real-space efficiency: {real_skip_reduction} skipped tests "
            "eliminated"
        )
        return
    elif space == Space.HARMONIC:  # pragma: no branch
        # Coverage.py artifact: branch-to-exit tracked but not separately testable
        harmonic_skip_reduction = len(harmonic_measurements) ** 2 - len(harmonic_pairs)
        console.print(
            f"\n🚀 Harmonic-space efficiency: {harmonic_skip_reduction} skipped "
            "tests eliminated"
        )
        return


if __name__ == "__main__":  # pragma: no cover
    app()
