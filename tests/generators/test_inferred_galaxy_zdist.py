"""Tests for the module firecrown.generators.inferred_galaxy_zdist."""

from typing import Any
from itertools import pairwise, product
import copy

import re
import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_equal, assert_allclose
from scipy.integrate import simpson
import yaml

from firecrown.generators.inferred_galaxy_zdist import (
    ZDistLSSTSRD,
    Y1_LENS_BINS,
    Y10_LENS_BINS,
    Y1_SOURCE_BINS,
    Y10_SOURCE_BINS,
    LinearGrid1D,
    ZDistLSSTSRDBin,
    ZDistLSSTSRDBinCollection,
    LSST_Y1_LENS_BIN_COLLECTION,
    LSST_Y1_SOURCE_BIN_COLLECTION,
    Measurement,
    make_measurements,
    make_measurements_dict,
)
from firecrown.metadata.two_point import Galaxies, Clusters, CMB
from firecrown.utils import base_model_from_yaml, base_model_to_yaml


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


BINS_LIST = ["one_lens", "all_lens", "one_source", "all_source", "lens_and_source"]


@pytest.fixture(
    name="zdist_bins",
    params=product(
        BINS_LIST,
        [True, False],
    ),
    ids=[f"{b[0]}_autoknots_{b[1]}" for b in product(BINS_LIST, [True, False])],
)
def fixture_zdist_bins(request) -> list[ZDistLSSTSRDBin]:
    """Fixture for the ZDistLSSTSRD class."""
    match request.param[0]:
        case "one_lens":
            bins = copy.deepcopy(LSST_Y1_LENS_BIN_COLLECTION.bins[0:1])
        case "all_lens":
            bins = copy.deepcopy(LSST_Y1_LENS_BIN_COLLECTION.bins)
        case "one_source":
            bins = copy.deepcopy(LSST_Y1_SOURCE_BIN_COLLECTION.bins[0:1])
        case "all_source":
            bins = copy.deepcopy(LSST_Y1_SOURCE_BIN_COLLECTION.bins)
        case "lens_and_source":
            bins = copy.deepcopy(LSST_Y1_LENS_BIN_COLLECTION.bins)
            bins.extend(copy.deepcopy(LSST_Y1_SOURCE_BIN_COLLECTION.bins))
        case _:
            raise ValueError(f"Invalid parameter: {request.param}")

    for zbin in bins:
        zbin.use_autoknot = request.param[1]

    return bins


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
        measurements={Galaxies.COUNTS},
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
            measurements={Galaxies.COUNTS},
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
        measurements={Galaxies.COUNTS},
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
            measurements={Galaxies.COUNTS},
            use_autoknot=True,
            autoknots_reltol=reltol,
        )

        assert_allclose(simpson(y=Pz.dndz, x=Pz.z), 1.0, atol=reltol)


def test_zdist_bin():
    """Test the ZDistLSSTSRDBin class."""
    zpl = 0.1
    zpu = 0.2
    sigma_z = 0.01
    z = LinearGrid1D(start=0.0, end=3.0, num=100)
    bin_name = "lens0_y1"
    measurements: set[Measurement] = {Galaxies.COUNTS}

    zbin = ZDistLSSTSRDBin(
        zpl=zpl,
        zpu=zpu,
        sigma_z=sigma_z,
        z=z,
        bin_name=bin_name,
        measurements=measurements,
    )

    assert zbin.zpl == zpl
    assert zbin.zpu == zpu
    assert zbin.sigma_z == sigma_z
    assert_array_equal(zbin.z.generate(), z.generate())
    assert zbin.bin_name == bin_name
    assert zbin.measurements == measurements


def test_zdist_bin_generate(zdist: ZDistLSSTSRD):
    """Test the ZDistLSSTSRDBin generate method."""
    zpl = 0.1
    zpu = 0.2
    sigma_z = 0.01
    z = LinearGrid1D(start=0.0, end=3.0, num=100)
    bin_name = "lens0_y1"
    measurements: set[Measurement] = {Galaxies.COUNTS}

    zbin = ZDistLSSTSRDBin(
        zpl=zpl,
        zpu=zpu,
        sigma_z=sigma_z,
        z=z,
        bin_name=bin_name,
        measurements=measurements,
    )

    Pz = zbin.generate(zdist)

    assert_array_equal(Pz.z, z.generate())
    assert Pz.bin_name == bin_name
    assert Pz.measurements == measurements


def test_zdist_bin_from_bad_yaml():
    """Test that the right exception is thrown if the yaml is malformed."""
    bin_yaml = """
    zpl: 0.1
    zpu: 0.2
    sigma_z: 0.01
    z:
        start: 0.0
        end: 3.0
        num: 100
    bin_name: lens0_y1
    measurement:
        subject: frogs
        property: COUNTS
    """

    with pytest.raises(ValueError, match="Error creating ZDistLSSTSRDBin from yaml"):
        _ = base_model_from_yaml(ZDistLSSTSRDBin, bin_yaml)


def test_zdist_bin_from_yaml():
    """Test the ZDistLSSTSRDBin class from_json method."""
    zpl = 0.1
    zpu = 0.2
    sigma_z = 0.01
    bin_name = "lens0_y1"
    measurements = {Galaxies.COUNTS}
    bin_yaml = """
    zpl: 0.1
    zpu: 0.2
    sigma_z: 0.01
    z:
        start: 0.0
        end: 3.0
        num: 100
    bin_name: lens0_y1
    measurements:
        - subject: Galaxies
          property: COUNTS
    """

    zbin = base_model_from_yaml(ZDistLSSTSRDBin, bin_yaml)

    assert zbin.zpl == zpl
    assert zbin.zpu == zpu
    assert zbin.sigma_z == sigma_z
    assert_array_equal(zbin.z.generate(), np.linspace(0.0, 3.0, 100))
    assert zbin.bin_name == bin_name
    assert zbin.measurements == measurements


def test_zdist_bin_to_yaml():
    """Test the ZDistLSSTSRDBin class to_yaml method."""
    zbin = ZDistLSSTSRDBin(
        zpl=0.1,
        zpu=0.2,
        sigma_z=0.01,
        z=LinearGrid1D(start=0.0, end=3.0, num=100),
        bin_name="lens0_y1",
        measurements={Galaxies.COUNTS},
    )
    assert isinstance(zbin, ZDistLSSTSRDBin)
    assert isinstance(zbin.z, LinearGrid1D)

    yaml_str = base_model_to_yaml(zbin)
    yaml_dict = yaml.safe_load(yaml_str)

    assert yaml_dict == {
        "zpl": zbin.zpl,
        "zpu": zbin.zpu,
        "sigma_z": zbin.sigma_z,
        "z": {"start": zbin.z.start, "end": zbin.z.end, "num": zbin.z.num},
        "bin_name": zbin.bin_name,
        "measurements": make_measurements_dict(zbin.measurements),
        "use_autoknot": zbin.use_autoknot,
        "autoknots_reltol": zbin.autoknots_reltol,
        "autoknots_abstol": zbin.autoknots_abstol,
    }


def test_zdist_bin_collection(zdist_bins):
    """Test the ZDistLSSTSRDBinCollection class."""
    alpha = 0.9
    beta = 2.0
    z0 = 0.5

    zbin_collection = ZDistLSSTSRDBinCollection(
        alpha=alpha, beta=beta, z0=z0, bins=zdist_bins
    )

    assert zbin_collection.alpha == alpha
    assert zbin_collection.beta == beta
    assert zbin_collection.z0 == z0
    assert zbin_collection.bins == zdist_bins


def test_zdist_bin_collection_generate(zdist_bins):
    """Test the ZDistLSSTSRDBinCollection generate method."""
    alpha = 0.9
    beta = 2.0
    z0 = 0.5

    zbin_collection = ZDistLSSTSRDBinCollection(
        alpha=alpha, beta=beta, z0=z0, bins=zdist_bins
    )

    Pz_list = zbin_collection.generate()

    assert zbin_collection.alpha == alpha
    assert zbin_collection.beta == beta
    assert zbin_collection.z0 == z0
    assert zbin_collection.bins == zdist_bins
    assert len(Pz_list) == len(zdist_bins)

    for zbin, Pz in zip(zdist_bins, Pz_list):
        if not zbin.use_autoknot:
            assert_array_equal(Pz.z, zbin.z.generate())
        else:
            assert not np.array_equal(Pz.z, zbin.z.generate())

        assert Pz.bin_name == zbin.bin_name
        assert Pz.measurements == zbin.measurements


def test_zdist_bin_collection_to_yaml(zdist_bins):
    """Test the ZDistLSSTSRDBinCollection to_yaml method."""
    alpha = 0.9
    beta = 2.0
    z0 = 0.5

    zbin_collection = ZDistLSSTSRDBinCollection(
        alpha=alpha, beta=beta, z0=z0, bins=zdist_bins
    )

    assert isinstance(zbin_collection, ZDistLSSTSRDBinCollection)

    yaml_str = base_model_to_yaml(zbin_collection)
    yaml_dict = yaml.safe_load(yaml_str)

    assert yaml_dict == {
        "alpha": alpha,
        "beta": beta,
        "z0": z0,
        "bins": [
            {
                "zpl": zbin.zpl,
                "zpu": zbin.zpu,
                "sigma_z": zbin.sigma_z,
                "z": zbin.z.model_dump(),
                "bin_name": zbin.bin_name,
                "measurements": make_measurements_dict(zbin.measurements),
                "use_autoknot": zbin.use_autoknot,
                "autoknots_reltol": zbin.autoknots_reltol,
                "autoknots_abstol": zbin.autoknots_abstol,
            }
            for zbin in zdist_bins
        ],
    }


def test_zdist_bin_collection_from_yaml(zdist_bins):
    """Test the ZDistLSSTSRDBinCollection from_yaml method."""
    alpha = 0.9
    beta = 2.0
    z0 = 0.5

    yaml_dict = {
        "alpha": alpha,
        "beta": beta,
        "z0": z0,
        "bins": [
            {
                "zpl": zbin.zpl,
                "zpu": zbin.zpu,
                "sigma_z": zbin.sigma_z,
                "z": zbin.z.model_dump(),
                "bin_name": zbin.bin_name,
                "measurements": make_measurements_dict(zbin.measurements),
                "use_autoknot": zbin.use_autoknot,
                "autoknots_reltol": zbin.autoknots_reltol,
                "autoknots_abstol": zbin.autoknots_abstol,
            }
            for zbin in zdist_bins
        ],
    }

    yaml_str = yaml.dump(yaml_dict)
    zbin_collection = base_model_from_yaml(ZDistLSSTSRDBinCollection, yaml_str)

    assert zbin_collection.alpha == alpha
    assert zbin_collection.beta == beta
    assert zbin_collection.z0 == z0
    assert len(zbin_collection.bins) == len(zdist_bins)
    for zbin, zbin_from_yaml in zip(zdist_bins, zbin_collection.bins):
        assert zbin.zpl == zbin_from_yaml.zpl
        assert zbin.zpu == zbin_from_yaml.zpu
        assert zbin.sigma_z == zbin_from_yaml.sigma_z
        assert_array_equal(zbin.z.generate(), zbin_from_yaml.z.generate())
        assert zbin.bin_name == zbin_from_yaml.bin_name
        assert zbin.measurements == zbin_from_yaml.measurements
        assert zbin.use_autoknot == zbin_from_yaml.use_autoknot
        assert zbin.autoknots_reltol == zbin_from_yaml.autoknots_reltol
        assert zbin.autoknots_abstol == zbin_from_yaml.autoknots_abstol


def test_make_measurement_from_measurement():
    cluster_meas: set[Measurement] = {Clusters.COUNTS}
    galaxy_meas: set[Measurement] = {Galaxies.SHEAR_E}
    cmb_meas: set[Measurement] = {CMB.CONVERGENCE}
    assert make_measurements(cluster_meas) == cluster_meas
    assert make_measurements(galaxy_meas) == galaxy_meas
    assert make_measurements(cmb_meas) == cmb_meas


def test_make_measurement_from_dictionary():
    cluster_info = [{"subject": "Clusters", "property": "COUNTS"}]
    galaxy_info = [{"subject": "Galaxies", "property": "SHEAR_E"}]
    cmb_info = [{"subject": "CMB", "property": "CONVERGENCE"}]

    assert make_measurements(cluster_info) == {Clusters.COUNTS}
    assert make_measurements(galaxy_info) == {Galaxies.SHEAR_E}
    assert make_measurements(cmb_info) == {CMB.CONVERGENCE}

    with pytest.raises(
        ValueError, match="Invalid Measurement: subject: 'frogs' is not recognized"
    ):
        _ = make_measurements([{"subject": "frogs", "property": "SHEAR_E"}])

    with pytest.raises(
        ValueError, match="Invalid Measurement: dictionary does not contain 'subject'"
    ):
        _ = make_measurements([{}])

    with pytest.raises(
        ValueError, match=re.escape(r"Invalid Measurement: {3} is not a dictionary")
    ):
        _ = make_measurements({3})  # type: ignore
