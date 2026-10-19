"""Tests for the module firecrown.generators.tomographic_bin."""

import copy
import math
import re
from itertools import pairwise, product

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from numcosmo_py import Ncm
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad, simpson
from scipy.optimize import brentq

from firecrown._source_arrays import _NBit, _RealArray
from firecrown.generators._inferred_galaxy_zdist import (
    BinsType,
    LSST_Y1_LENS_HARMONIC_BIN_COLLECTION,
    LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION,
    LSST_Y10_LENS_HARMONIC_BIN_COLLECTION,
    LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION,
    Y1_LENS_BINS,
    Y1_SOURCE_BINS,
    Y10_LENS_BINS,
    Y10_SOURCE_BINS,
    LinearGrid1D,
    RawGrid1D,
    Measurement,
    ZDistLSSTSRD,
    ZDistLSSTSRDBin,
    ZDistLSSTSRDBinCollection,
    make_measurements,
    make_measurements_dict,
)
from firecrown.metadata_types import CMB, Clusters, Galaxies
from firecrown.utils import base_model_from_yaml, base_model_to_yaml


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_distribution_real_precision(
    dtype: type[np.integer[_NBit] | np.floating[_NBit]],
) -> None:
    """Evaluate real coordinates at double precision before returning values.

    :param dtype: Supported coordinate storage supplied by the caller.
    """
    z: _RealArray[_NBit] = np.array([0, 1, 2, 3], dtype=dtype)
    # For alpha=1, beta=2, z0=1 the normalized density is z² exp(-z)/2.
    expected = [0.0, math.exp(-1) / 2, 2 * math.exp(-2), 4.5 * math.exp(-3)]
    result = ZDistLSSTSRD(alpha=1.0, beta=2.0, z0=1.0).distribution(z)
    assert result.dtype == np.float64
    assert_allclose(result, expected, rtol=2e-15, atol=0.0)


@pytest.mark.parametrize("use_autoknot", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_binned_distribution_real_precision(
    use_autoknot: bool,
    dtype: type[np.integer[_NBit] | np.floating[_NBit]],
) -> None:
    """Preserve Gaussian-bin probabilities and normalization at double precision.

    :param use_autoknot: Select the existing spline or direct branch.
    :param dtype: Supported coordinate representation.
    """
    z: _RealArray[_NBit] = np.array([0, 1, 2, 3], dtype=dtype)
    zdist = ZDistLSSTSRD(
        alpha=1.0,
        beta=2.0,
        z0=1.0,
        use_autoknot=use_autoknot,
        autoknots_reltol=1e-5,
    )
    result = zdist.binned_distribution(
        zpl=0.6,
        zpu=1.4,
        sigma_z=0.2,
        z=z,
        name="precision",
        measurements={Galaxies.COUNTS},
    )

    def density(value: float) -> float:
        width = math.sqrt(2) * 0.2 * (1 + value)
        probability = (
            math.erf((1.4 - value) / width) - math.erf((0.6 - value) / width)
        ) / (1 + math.erf(value / width))
        return value**2 * math.exp(-value) / 2 * probability

    normalization = quad(lambda value: density(value) + 1e-15, 0.0, 3.0)[0]
    expected = np.array([density(float(value)) for value in result.z], dtype=np.float64)
    if use_autoknot:
        expected += 1e-15
    expected /= normalization
    assert result.z.dtype == np.float64
    assert result.dndz.dtype == np.float64
    assert_allclose(result.dndz, expected, rtol=2e-4 if use_autoknot else 2e-13)
    if not use_autoknot:
        assert_array_equal(result.z, [0.0, 1.0, 2.0, 3.0])
        if dtype is np.float64:
            assert np.shares_memory(result.z, z)


@pytest.mark.parametrize(
    "z",
    [
        np.array([False, True]),
        np.array([0.0, 1.0], dtype=np.complex128),
        np.array(["0", "1"]),
        np.array([0.0, 1.0], dtype=object),
        np.array([(0.0,), (1.0,)], dtype=[("z", np.float64)]),
        np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        np.array([0, 1], dtype="timedelta64[D]"),
        np.ma.array([0.0, 1.0], mask=[False, True]),
        np.ma.array([0.0, 1.0], mask=False),
    ],
    ids=[
        "bool",
        "complex",
        "string",
        "object",
        "structured",
        "datetime",
        "timedelta",
        "masked",
        "empty-mask",
    ],
)
@pytest.mark.parametrize("ingress", ["distribution", "direct-bin", "autoknot-bin"])
def test_generator_rejects_unsupported_arrays(
    z: npt.NDArray[np.generic],
    ingress: str,
) -> None:
    """Reject physical representations before computation can lose information.

    :param z: Deliberately unsupported coordinate storage.
    :param ingress: Public distribution or binned-generation boundary.
    """
    zdist = ZDistLSSTSRD.year_1_lens(use_autoknot=ingress == "autoknot-bin")
    with pytest.raises(TypeError, match="real integer or floating|masked"):
        if ingress == "distribution":
            zdist.distribution(z)  # type: ignore[arg-type]  # Invalid input.
        else:
            zdist.binned_distribution(
                zpl=0.6,
                zpu=1.4,
                sigma_z=0.2,
                z=z,  # type: ignore[arg-type]  # Invalid input.
                name="invalid",
                measurements={Galaxies.COUNTS},
            )


@pytest.mark.parametrize("num", [0, 1, 7])
def test_linear_grid_float32_endpoints(num: int) -> None:
    """Calculate grids from the supplied representable endpoints in float64.

    :param num: Existing supported empty, singleton, and ordinary grid lengths.
    """
    start, end = np.float32(0.01), np.float32(1.7)
    # Pydantic supports these real scalar inputs and converts its float fields.
    grid = LinearGrid1D(start=start, end=end, num=num)  # type: ignore[arg-type]
    assert type(grid.start) is float
    assert type(grid.end) is float
    lower, upper = float(start), float(end)
    expected = np.array(
        (
            [lower + (upper - lower) * i / (num - 1) for i in range(num)]
            if num > 1
            else [lower] * num
        ),
        dtype=np.float64,
    )
    result = grid.generate()
    assert result.dtype == np.float64
    assert_allclose(result, expected, rtol=0.0, atol=3e-16)


@pytest.mark.parametrize("values", [[], [0.0], [0.0, 1.0, 2.0]])
def test_raw_grid_float64(values: list[float]) -> None:
    """Generate explicit real storage, including established empty results.

    :param values: Raw coordinate values.
    """
    result = RawGrid1D(values=values).generate()
    assert result.dtype == np.float64
    assert_array_equal(result, values)


def test_distribution_scalar_and_empty() -> None:
    """Retain scalar outputs and empty array shape with float64 values."""
    zdist = ZDistLSSTSRD(alpha=1.0, beta=2.0, z0=1.0)
    scalar = zdist.distribution(1.0)
    assert isinstance(scalar, float)
    assert_allclose(scalar, math.exp(-1) / 2, rtol=2e-15)
    result = zdist.distribution(np.empty((0, 2), dtype=np.float32))
    assert result.dtype == np.float64
    assert result.shape == (0, 2)


@pytest.mark.parametrize("use_autoknot", [False, True])
def test_binned_distribution_empty_remains_invalid(use_autoknot: bool) -> None:
    """Keep the established requirement for integration-bound coordinates.

    :param use_autoknot: Existing direct or spline branch.
    """
    zdist = ZDistLSSTSRD.year_1_lens(use_autoknot=use_autoknot)
    with pytest.raises(IndexError):
        zdist.binned_distribution(
            zpl=0.6,
            zpu=1.4,
            sigma_z=0.2,
            z=np.empty(0, dtype=np.float64),
            name="empty",
            measurements={Galaxies.COUNTS},
        )


@pytest.mark.parametrize("use_true_distribution", [False, True])
def test_equal_area_edges_float64(use_true_distribution: bool) -> None:
    """Compare NumCosmo bin edges with independent analytic CDF integration.

    :param use_true_distribution: Bin the true or convolved distribution.
    """
    zdist = ZDistLSSTSRD(
        alpha=1.0,
        beta=2.0,
        z0=1.0,
        max_z=5.0,
        autoknots_reltol=1e-6,
    )

    def cdf(limit: float) -> float:
        if use_true_distribution:
            return 1 - math.exp(-limit) * (1 + limit + limit**2 / 2)

        def integrand(value: float) -> float:
            width = math.sqrt(2) * 0.2 * (1 + value)
            probability = (
                math.erf((limit - value) / width) + math.erf(value / width)
            ) / (1 + math.erf(value / width))
            return value**2 * math.exp(-value) / 2 * probability

        return quad(integrand, 0.0, 5.0)[0]

    half_area = cdf(3.0) / 2
    middle = brentq(lambda limit: cdf(limit) - half_area, 0.0, 3.0)
    edges = zdist.equal_area_bins(
        2,
        0.2,
        3.0,
        use_true_distribution=use_true_distribution,
    )
    assert edges.dtype == np.float64
    assert_allclose(edges, [0.0, middle, 3.0], rtol=1e-5, atol=1e-7)


@pytest.fixture(
    name="zdist",
    params=product(
        [
            ZDistLSSTSRD.year_1_lens,
            ZDistLSSTSRD.year_10_lens,
            ZDistLSSTSRD.year_1_source,
            ZDistLSSTSRD.year_10_source,
        ],
        [True, False],
        [1e-4, 1e-5],
    ),
    ids=[
        f"{dist}-autoknots-{use_autoknots}-reltol-{reltol}"
        for dist, use_autoknots, reltol in product(
            [
                "Y1_lens",
                "Y10_lens",
                "Y1_source",
                "Y10_source",
            ],
            ["true", "false"],
            [1e-4, 1e-5],
        )
    ],
)
def fixture_zdist_y1(request: pytest.FixtureRequest) -> ZDistLSSTSRD:
    """Fixture for the ZDistLSSTSRD class."""
    return request.param[0](
        use_autoknot=request.param[1], autoknots_reltol=request.param[2]
    )


@pytest.fixture(name="z_array", params=[100, 600, 1000])
def fixture_z_array(request: pytest.FixtureRequest) -> npt.NDArray[np.float64]:
    """Fixture for the z array."""
    return np.linspace(0.0, 3.0, request.param, dtype=np.float64)


@pytest.fixture(
    name="bins", params=[Y1_LENS_BINS, Y10_LENS_BINS, Y1_SOURCE_BINS, Y10_SOURCE_BINS]
)
def fixture_bins(request: pytest.FixtureRequest) -> BinsType:
    """Fixture for the bins."""
    return request.param


@pytest.fixture(name="reltol", params=[1e-4, 1e-5])
def fixture_reltol(request: pytest.FixtureRequest) -> float:
    """Fixture for the relative tolerance."""
    return request.param


BINS_LIST = [
    "one_lens_y1",
    "all_lens_y1",
    "one_source_y1",
    "all_source_y1",
    "lens_and_source_y1",
    "one_lens_y10",
    "all_lens_y10",
    "one_source_y10",
    "all_source_y10",
    "lens_and_source_y10",
]


@pytest.fixture(
    name="zdist_bins",
    params=product(
        BINS_LIST,
        [True, False],
    ),
    ids=[f"{b[0]}_autoknots_{b[1]}" for b in product(BINS_LIST, [True, False])],
)
def fixture_zdist_bins(
    request: pytest.FixtureRequest,
) -> list[ZDistLSSTSRDBin]:
    """Fixture for the ZDistLSSTSRD class."""
    match request.param[0]:
        case "one_lens_y1":
            bins = copy.deepcopy(LSST_Y1_LENS_HARMONIC_BIN_COLLECTION.bins[0:1])
        case "all_lens_y1":
            bins = copy.deepcopy(LSST_Y1_LENS_HARMONIC_BIN_COLLECTION.bins)
        case "one_source_y1":
            bins = copy.deepcopy(LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION.bins[0:1])
        case "all_source_y1":
            bins = copy.deepcopy(LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION.bins)
        case "lens_and_source_y1":
            bins = copy.deepcopy(LSST_Y1_LENS_HARMONIC_BIN_COLLECTION.bins)
            bins.extend(copy.deepcopy(LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION.bins))
        case "one_lens_y10":
            bins = copy.deepcopy(LSST_Y10_LENS_HARMONIC_BIN_COLLECTION.bins[0:1])
        case "all_lens_y10":
            bins = copy.deepcopy(LSST_Y10_LENS_HARMONIC_BIN_COLLECTION.bins)
        case "one_source_y10":
            bins = copy.deepcopy(LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION.bins[0:1])
        case "all_source_y10":
            bins = copy.deepcopy(LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION.bins)
        case "lens_and_source_y10":
            bins = copy.deepcopy(LSST_Y10_LENS_HARMONIC_BIN_COLLECTION.bins)
            bins.extend(copy.deepcopy(LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION.bins))
        case _:
            raise ValueError(f"Invalid parameter: {request.param}")

    return bins


def test_zdist(zdist: ZDistLSSTSRD, z_array: npt.NDArray[np.float64]) -> None:
    """Test the ZDistLSSTSRD class."""
    Pz = zdist.distribution(z_array)
    assert Pz.shape == z_array.shape
    assert Pz.dtype == np.float64


def test_compute_one_bin_dist(
    zdist: ZDistLSSTSRD,
    z_array: npt.NDArray[np.float64],
    bins: BinsType,
) -> None:
    """Test the compute_binned_dist method."""

    Pz = zdist.binned_distribution(
        zpl=bins["edges"][0],
        zpu=bins["edges"][1],
        sigma_z=bins["sigma_z"],
        z=z_array,
        name="lens0_y1",
        measurements={Galaxies.COUNTS},
    )

    if not zdist.use_autoknot:
        assert_array_equal(Pz.z, z_array)
    assert Pz.z.dtype == np.float64
    assert Pz.dndz.dtype == np.float64
    assert bins["edges"].dtype == np.float64
    assert_allclose(simpson(y=Pz.dndz, x=Pz.z), 1.0, atol=1e-3)


def test_compute_all_bins_dist(
    zdist: ZDistLSSTSRD,
    z_array: npt.NDArray[np.float64],
    bins: BinsType,
) -> None:
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
        if not zdist.use_autoknot:
            assert_array_equal(Pz.z, z_array)
        assert_allclose(simpson(y=Pz.dndz, x=Pz.z), 1.0, atol=1e-3)


def test_zdist_bin() -> None:
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


def test_zdist_bin_generate(zdist: ZDistLSSTSRD) -> None:
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

    if not zdist.use_autoknot:
        assert_array_equal(Pz.z, z.generate())
    assert Pz.bin_name == bin_name
    assert Pz.measurements == measurements


def test_zdist_bin_from_bad_yaml() -> None:
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


def test_zdist_bin_from_yaml() -> None:
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
    assert_array_equal(zbin.z.generate(), np.linspace(0.0, 3.0, 100, dtype=np.float64))
    assert zbin.bin_name == bin_name
    assert zbin.measurements == measurements


def test_zdist_bin_to_yaml() -> None:
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
    }


def test_zdist_bin_collection(zdist_bins: list[ZDistLSSTSRDBin]) -> None:
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


def test_zdist_bin_collection_generate(
    zdist_bins: list[ZDistLSSTSRDBin],
) -> None:
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
        if not zbin_collection.use_autoknot:
            assert_array_equal(Pz.z, zbin.z.generate())
        else:
            assert not np.array_equal(Pz.z, zbin.z.generate())

        assert Pz.bin_name == zbin.bin_name
        assert Pz.measurements == zbin.measurements


def test_zdist_bin_collection_to_yaml(
    zdist_bins: list[ZDistLSSTSRDBin],
) -> None:
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
        "max_z": zbin_collection.max_z,
        "use_autoknot": zbin_collection.use_autoknot,
        "autoknots_reltol": zbin_collection.autoknots_reltol,
        "autoknots_abstol": zbin_collection.autoknots_abstol,
        "bins": [
            {
                "zpl": zbin.zpl,
                "zpu": zbin.zpu,
                "sigma_z": zbin.sigma_z,
                "z": zbin.z.model_dump(),
                "bin_name": zbin.bin_name,
                "measurements": make_measurements_dict(zbin.measurements),
            }
            for zbin in zdist_bins
        ],
    }


def test_zdist_bin_collection_from_yaml(
    zdist_bins: list[ZDistLSSTSRDBin],
) -> None:
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


def test_make_measurement_from_measurement() -> None:
    cluster_meas: set[Measurement] = {Clusters.COUNTS}
    galaxy_meas: set[Measurement] = {Galaxies.SHEAR_E}
    cmb_meas: set[Measurement] = {CMB.CONVERGENCE}
    assert make_measurements(cluster_meas) == cluster_meas
    assert make_measurements(galaxy_meas) == galaxy_meas
    assert make_measurements(cmb_meas) == cmb_meas


def test_make_measurement_from_dictionary() -> None:
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
        ValueError, match=re.escape(r"Invalid Measurement: 3 is not a dictionary")
    ):
        _ = make_measurements({3})  # type: ignore


def test_distribution(zdist: ZDistLSSTSRD) -> None:
    """Test the distribution method."""

    stats = zdist.compute_distribution(0.03)
    assert isinstance(stats, Ncm.StatsDist1d)

    middle_z = stats.eval_inv_pdf(0.5)

    bins = zdist.equal_area_bins(2, 0.03, zdist.max_z)

    assert_allclose([0.0, middle_z, zdist.max_z], bins)


def test_true_distribution(zdist: ZDistLSSTSRD) -> None:
    """Test the true_distribution method."""
    stats = zdist.compute_true_distribution()
    assert isinstance(stats, Ncm.StatsDist1d)

    middle_z = stats.eval_inv_pdf(0.5)

    bins = zdist.equal_area_bins(2, 0.03, zdist.max_z, use_true_distribution=True)

    assert_allclose([0.0, middle_z, zdist.max_z], bins)
