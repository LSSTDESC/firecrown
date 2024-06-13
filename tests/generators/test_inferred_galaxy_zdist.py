"""Tests for the module firecrown.generators.inferred_galaxy_zdist."""

from typing import Any
from itertools import pairwise

import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_equal, assert_allclose
from scipy.integrate import simpson

from firecrown.generators.inferred_galaxy_zdist import (
    ZDistLSSTSRD,
    Y1_LENS_BINS,
    Y10_LENS_BINS,
    Y1_SOURCE_BINS,
    Y10_SOURCE_BINS,
)
from firecrown.metadata.two_point import Galaxies


@pytest.fixture(name="zdist", params=[ZDistLSSTSRD.year_1(), ZDistLSSTSRD.year_10()])
def fixture_zdist_y1(request):
    """Fixture for the ZDistLSSTSRD class."""
    return request.param


@pytest.fixture(name="z_array", params=[100, 600, 1000])
def fixture_z_array(request):
    """Fixture for the z array."""
    return np.linspace(0, 3.0, request.param)


@pytest.fixture(
    name="bins", params=[Y1_LENS_BINS, Y10_LENS_BINS, Y1_SOURCE_BINS, Y10_SOURCE_BINS]
)
def fixture_bins(request) -> dict[str, Any]:
    """Fixture for the bins."""
    return request.param


@pytest.fixture(name="reltol", params=[1e-4, 1e-5])
def fixture_reltol(request):
    """Fixture for the relative tolerance."""
    return request.param


def test_zdist(zdist: ZDistLSSTSRD, z_array: npt.NDArray[np.float64]):
    """Test the ZDistLSSTSRD class."""
    Pz = zdist.distribution(z_array)
    assert Pz.shape == z_array.shape


def test_compute_one_bin_dist_fix_z(
    zdist: ZDistLSSTSRD,
    z_array: npt.NDArray[np.float64],
    bins: dict[str, Any],
):
    """Test the compute_binned_dist method."""

    Pz = zdist.binned_distribution(
        zpl=bins["edges"][0],
        zpu=bins["edges"][1],
        sigma_z=bins["sigma_z"],
        z=z_array,
        name="lens0_y1",
        measurement=Galaxies.COUNTS,
    )

    assert_array_equal(Pz.z, z_array)
    assert_allclose(simpson(y=Pz.dndz, x=z_array), 1.0, atol=1e-3)


def test_compute_all_bins_dist_fix_z(
    zdist: ZDistLSSTSRD,
    z_array: npt.NDArray[np.float64],
    bins: dict[str, Any],
):
    """Test the compute_binned_dist method."""

    for zpl, zpu in pairwise(bins["edges"]):
        Pz = zdist.binned_distribution(
            zpl=zpl,
            zpu=zpu,
            sigma_z=bins["sigma_z"],
            z=z_array,
            name="lens_y1",
            measurement=Galaxies.COUNTS,
        )

        assert_array_equal(Pz.z, z_array)
        assert_allclose(simpson(y=Pz.dndz, x=z_array), 1.0, atol=1e-3)


def test_compute_one_bin_dist_autoknot(
    zdist: ZDistLSSTSRD,
    z_array: npt.NDArray[np.float64],
    bins: dict[str, Any],
    reltol: float,
):
    """Test the compute_binned_dist method."""

    Pz = zdist.binned_distribution(
        zpl=bins["edges"][0],
        zpu=bins["edges"][1],
        sigma_z=bins["sigma_z"],
        z=z_array,
        name="lens0_y1",
        measurement=Galaxies.COUNTS,
        use_autoknot=True,
        autoknots_reltol=reltol,
    )

    assert_allclose(simpson(y=Pz.dndz, x=Pz.z), 1.0, atol=reltol)


def test_compute_all_bins_dist_autoknot(
    zdist: ZDistLSSTSRD,
    z_array: npt.NDArray[np.float64],
    bins: dict[str, Any],
    reltol: float,
):
    """Test the compute_binned_dist method."""

    for zpl, zpu in pairwise(bins["edges"]):
        Pz = zdist.binned_distribution(
            zpl=zpl,
            zpu=zpu,
            sigma_z=bins["sigma_z"],
            z=z_array,
            name="lens_y1",
            measurement=Galaxies.COUNTS,
            use_autoknot=True,
            autoknots_reltol=reltol,
        )

        assert_allclose(simpson(y=Pz.dndz, x=Pz.z), 1.0, atol=reltol)
