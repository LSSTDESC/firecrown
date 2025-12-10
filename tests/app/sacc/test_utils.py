"""Unit tests for firecrown.app.sacc._utils module.

Tests for utility functions used in SACC operations.
"""

import pytest
import numpy as np
from firecrown import metadata_types as mdt
from firecrown.app.sacc._utils import mean_std_tracer


@pytest.fixture(name="mock_tracer")
def fixture_mock_tracer() -> mdt.InferredGalaxyZDist:
    """Create mock tracer with Gaussian distribution."""
    z = np.linspace(0.0, 2.0, 100)
    mean = 1.0
    sigma = 0.2
    dndz = np.exp(-0.5 * ((z - mean) / sigma) ** 2)
    dndz /= np.trapezoid(dndz, z)

    return mdt.InferredGalaxyZDist(
        bin_name="bin0",
        z=z,
        dndz=dndz,
        measurements=set([mdt.Galaxies.COUNTS]),
        type_source=mdt.TypeSource("firecrown"),
    )


class TestMeanStdTracer:
    """Tests for mean_std_tracer function."""

    def test_mean_std_tracer_gaussian(
        self, mock_tracer: mdt.InferredGalaxyZDist
    ) -> None:
        """Test mean_std_tracer with Gaussian distribution."""
        mean, std = mean_std_tracer(mock_tracer)

        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert 0.9 < mean < 1.1  # Expected mean ~1.0
        assert 0.15 < std < 0.25  # Expected std ~0.2

    def test_mean_std_tracer_uniform(self) -> None:
        """Test mean_std_tracer with uniform distribution."""
        z = np.linspace(0.5, 1.5, 100)
        dndz = np.ones_like(z)
        dndz /= np.trapezoid(dndz, z)

        tracer = mdt.InferredGalaxyZDist(
            bin_name="uniform",
            z=z,
            dndz=dndz,
            measurements=set([mdt.Galaxies.COUNTS]),
            type_source=mdt.TypeSource("test"),
        )

        mean, std = mean_std_tracer(tracer)

        assert 0.95 < mean < 1.05  # Expected mean = 1.0
        assert 0.25 < std < 0.35  # Expected std ~0.289

    def test_mean_std_tracer_delta_function(self) -> None:
        """Test mean_std_tracer with delta-like distribution."""
        z = np.linspace(0.0, 2.0, 100)
        dndz = np.zeros_like(z)
        center_idx = 50
        dndz[center_idx] = 1.0
        dndz /= np.trapezoid(dndz, z)

        tracer = mdt.InferredGalaxyZDist(
            bin_name="delta",
            z=z,
            dndz=dndz,
            measurements=set([mdt.Galaxies.COUNTS]),
            type_source=mdt.TypeSource("test"),
        )

        mean, std = mean_std_tracer(tracer)

        # Mean should be approximately at the delta function location
        assert 0.9 < mean < 1.1
        # Std should be very small for a delta-like function
        assert std < 0.1

    def test_mean_std_tracer_skewed_distribution(self) -> None:
        """Test mean_std_tracer with skewed distribution."""
        z = np.linspace(0.0, 2.0, 100)
        # Create a skewed distribution with more weight at higher z
        dndz = np.exp(-0.5 * ((z - 1.3) / 0.3) ** 2)
        dndz /= np.trapezoid(dndz, z)

        tracer = mdt.InferredGalaxyZDist(
            bin_name="skewed",
            z=z,
            dndz=dndz,
            measurements=set([mdt.Galaxies.COUNTS]),
            type_source=mdt.TypeSource("test"),
        )

        mean, std = mean_std_tracer(tracer)

        # Mean should be shifted towards higher z
        assert 1.1 < mean < 1.5
        # Std should be reasonable
        assert 0.15 < std < 0.35
