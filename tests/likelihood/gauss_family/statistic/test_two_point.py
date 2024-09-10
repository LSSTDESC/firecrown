"""
Tests for the TwoPoint module.
"""

import re
from unittest.mock import MagicMock
import numpy as np
from numpy.testing import assert_allclose
import pytest

import pyccl

from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap

from firecrown.likelihood.number_counts import (
    NumberCounts,
)
from firecrown.likelihood.weak_lensing import (
    WeakLensing,
)
from firecrown.likelihood.statistic import TheoryVector
from firecrown.likelihood.two_point import (
    _ell_for_xi,
    TwoPoint,
    TracerNames,
    TRACER_NAMES_TOTAL,
    EllOrThetaConfig,
    use_source_factory,
    use_source_factory_metadata_index,
    WeakLensingFactory,
    NumberCountsFactory,
)
from firecrown.metadata_types import (
    Galaxies,
    InferredGalaxyZDist,
    TwoPointHarmonic,
    GALAXY_LENS_TYPES,
    GALAXY_SOURCE_TYPES,
)
from firecrown.metadata_functions import (
    TwoPointHarmonicIndex,
    TwoPointRealIndex,
)
from firecrown.data_types import TwoPointMeasurement


@pytest.fixture(name="source_0")
def fixture_source_0() -> NumberCounts:
    """Return an almost-default NumberCounts source."""
    return NumberCounts(sacc_tracer="lens_0")


@pytest.fixture(name="tools")
def fixture_tools() -> ModelingTools:
    """Return a trivial ModelingTools object."""
    return ModelingTools()


@pytest.fixture(name="harmonic_data_with_window")
def fixture_harmonic_data_with_window(harmonic_two_point_xy) -> TwoPointMeasurement:
    """Return some fake harmonic data."""
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    # The window is given by the mean of the ells in each bin times the bin number.
    weights = np.zeros((100, 4))
    weights[0:25, 0] = 1.0 / 25.0
    weights[25:50, 1] = 2.0 / 25.0
    weights[50:75, 2] = 3.0 / 25.0
    weights[75:100, 3] = 4.0 / 25.0

    data = np.zeros(4) + 1.1
    indices = np.arange(4)
    covariance_name = "cov"
    tpm = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointHarmonic(ells=ells, window=weights, XY=harmonic_two_point_xy),
    )

    return tpm


@pytest.fixture(name="harmonic_data_no_window")
def fixture_harmonic_data_no_window(harmonic_two_point_xy) -> TwoPointMeasurement:
    """Return some fake harmonic data."""
    ells = np.array(np.linspace(0, 100, 100), dtype=np.int64)
    data = np.zeros(100) - 1.1
    indices = np.arange(100)
    covariance_name = "cov"
    tpm = TwoPointMeasurement(
        data=data,
        indices=indices,
        covariance_name=covariance_name,
        metadata=TwoPointHarmonic(ells=ells, XY=harmonic_two_point_xy),
    )

    return tpm


def test_ell_for_xi_no_rounding():
    res = _ell_for_xi(minimum=0, midpoint=5, maximum=80, n_log=5)
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    assert res.shape == expected.shape
    assert np.allclose(expected, res)


def test_ell_for_xi_doing_rounding():
    res = _ell_for_xi(minimum=1, midpoint=3, maximum=100, n_log=5)
    expected = np.array([1.0, 2.0, 3.0, 7.0, 17.0, 42.0, 100.0])
    assert np.allclose(expected, res)


def test_compute_theory_vector(source_0: NumberCounts):
    # To create the TwoPoint object we need at least one source.
    statistic = TwoPoint("galaxy_density_xi", source_0, source_0)
    assert isinstance(statistic, TwoPoint)

    # Before calling compute_theory_vector, we must get the TwoPoint object
    # into the correct state.
    # prediction = statistic.compute_theory_vector(tools)
    # assert isinstance(prediction, TheoryVector)


def test_tracer_names():
    assert TracerNames("", "") == TRACER_NAMES_TOTAL

    tn1 = TracerNames("cow", "pig")
    assert tn1[0] == "cow"
    assert tn1[1] == "pig"

    tn2 = TracerNames("cat", "dog")
    assert tn1 != tn2
    assert hash(tn1) != hash(tn2)

    with pytest.raises(IndexError):
        _ = tn1[2]


def test_two_point_src0_src0_window(sacc_galaxy_cells_src0_src0_window):
    """This test also makes sure that TwoPoint theory calculations are
    repeatable."""
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_window

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = TwoPoint("galaxy_shear_cl_ee", src0, src0)
    statistic.read(sacc_data)

    tools = ModelingTools()
    tools.update(ParamsMap())
    tools.prepare(pyccl.CosmologyVanillaLCDM())

    assert statistic.window is not None

    statistic.reset()
    statistic.update(ParamsMap())
    tools.update(ParamsMap())
    result1 = statistic.compute_theory_vector(tools)
    assert all(np.isfinite(result1))

    statistic.reset()
    statistic.update(ParamsMap())
    tools.update(ParamsMap())
    result2 = statistic.compute_theory_vector(tools)
    assert np.array_equal(result1, result2)


def test_two_point_src0_src0_no_window(sacc_galaxy_cells_src0_src0_no_window):
    """This test also makes sure that TwoPoint theory calculations are
    repeatable."""
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_window

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = TwoPoint("galaxy_shear_cl_ee", src0, src0)
    statistic.read(sacc_data)

    tools = ModelingTools()
    tools.update(ParamsMap())
    tools.prepare(pyccl.CosmologyVanillaLCDM())

    assert statistic.window is None

    statistic.reset()
    statistic.update(ParamsMap())
    tools.update(ParamsMap())
    result1 = statistic.compute_theory_vector(tools)
    assert all(np.isfinite(result1))

    statistic.reset()
    statistic.update(ParamsMap())
    tools.update(ParamsMap())
    result2 = statistic.compute_theory_vector(tools)
    assert np.array_equal(result1, result2)


def test_two_point_src0_src0_no_data_lin(sacc_galaxy_cells_src0_src0_no_data):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_data

    src0 = WeakLensing(sacc_tracer="src0")

    ell_config: EllOrThetaConfig = {
        "minimum": 1,
        "maximum": 100,
        "n": 5,
        "binning": "lin",
    }

    statistic = TwoPoint(
        "galaxy_shear_cl_ee",
        src0,
        src0,
        ell_or_theta=ell_config,
    )
    with pytest.warns(UserWarning, match="Empty index selected"):
        statistic.read(sacc_data)

    assert statistic.window is None
    assert statistic.ells is not None
    assert statistic.thetas is None
    assert all(np.isfinite(statistic.ells))
    assert all(statistic.ells >= 1)
    assert all(statistic.ells <= 100)


def test_two_point_src0_src0_no_data_log(sacc_galaxy_cells_src0_src0_no_data):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_data

    src0 = WeakLensing(sacc_tracer="src0")

    ell_config: EllOrThetaConfig = {
        "minimum": 1,
        "maximum": 100,
        "n": 5,
        "binning": "log",
    }

    statistic = TwoPoint(
        "galaxy_shear_cl_ee",
        src0,
        src0,
        ell_or_theta=ell_config,
    )
    with pytest.warns(UserWarning, match="Empty index selected"):
        statistic.read(sacc_data)

    assert statistic.window is None
    assert statistic.ells is not None
    assert statistic.thetas is None
    assert all(np.isfinite(statistic.ells))
    assert all(statistic.ells >= 1)
    assert all(statistic.ells <= 100)


def test_two_point_lens0_lens0_no_data(sacc_galaxy_xis_lens0_lens0_no_data):
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0_no_data

    src0 = NumberCounts(sacc_tracer="lens0")

    ell_config: EllOrThetaConfig = {
        "minimum": 0.0,
        "maximum": 1.0,
        "n": 5,
        "binning": "lin",
    }

    statistic = TwoPoint(
        "galaxy_density_xi",
        src0,
        src0,
        ell_or_theta=ell_config,
    )
    with pytest.warns(UserWarning):
        statistic.read(sacc_data)

    assert statistic.window is None
    assert statistic.ells is None
    assert statistic.thetas is not None
    assert all(np.isfinite(statistic.thetas))
    assert all(statistic.thetas >= 0.0)
    assert all(statistic.thetas <= 1.0)


def test_two_point_src0_src0_cuts(sacc_galaxy_cells_src0_src0):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = TwoPoint(
        "galaxy_shear_cl_ee", src0, src0, ell_or_theta_min=50, ell_or_theta_max=200
    )
    with pytest.warns(
        UserWarning, match="No bandpower windows associated to these data"
    ):
        statistic.read(sacc_data)

    tools = ModelingTools()
    tools.update(ParamsMap())
    tools.prepare(pyccl.CosmologyVanillaLCDM())

    assert statistic.window is None
    assert statistic.ells is not None
    assert all(np.isfinite(statistic.ells))
    assert all(statistic.ells >= 50)
    assert all(statistic.ells <= 200)

    statistic.update(ParamsMap())
    statistic.compute_theory_vector(tools)


def test_two_point_lens0_lens0_cuts(sacc_galaxy_xis_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0

    src0 = NumberCounts(sacc_tracer="lens0")

    statistic = TwoPoint(
        "galaxy_density_xi", src0, src0, ell_or_theta_min=0.1, ell_or_theta_max=0.5
    )
    statistic.read(sacc_data)

    param_map = ParamsMap({"lens0_bias": 1.0})
    tools = ModelingTools()
    tools.update(param_map)
    tools.prepare(pyccl.CosmologyVanillaLCDM())

    assert statistic.window is None
    assert statistic.thetas is not None
    assert all(np.isfinite(statistic.thetas))
    assert all(statistic.thetas >= 0.1)
    assert all(statistic.thetas <= 0.5)

    statistic.update(param_map)
    statistic.compute_theory_vector(tools)


def test_two_point_lens0_lens0_config(sacc_galaxy_xis_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0

    src0 = NumberCounts(sacc_tracer="lens0")

    statistic = TwoPoint(
        "galaxy_density_xi",
        src0,
        src0,
        ell_for_xi={"minimum": 2, "midpoint": 6, "n_log": 180},
    )
    statistic.read(sacc_data)

    param_map = ParamsMap({"lens0_bias": 1.0})
    tools = ModelingTools()
    tools.update(param_map)
    tools.prepare(pyccl.CosmologyVanillaLCDM())

    assert statistic.window is None
    assert statistic.thetas is not None
    assert all(np.isfinite(statistic.thetas))
    assert statistic.ells_for_xi is not None
    assert all(np.isfinite(statistic.ells_for_xi))
    # The size depends on the configuration but also
    # on how many unique ells we get from the log-binning.
    assert len(statistic.ells_for_xi) == 175

    statistic.update(param_map)
    statistic.compute_theory_vector(tools)


def test_two_point_src0_src0_no_data_error(sacc_galaxy_cells_src0_src0_no_data):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_data

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = TwoPoint("galaxy_shear_cl_ee", src0, src0)
    with pytest.warns(UserWarning, match="Empty index selected"):
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "for data type 'galaxy_shear_cl_ee' have no 2pt data in the SACC file "
                "and no input ell values were given"
            ),
        ):
            statistic.read(sacc_data)


def test_two_point_lens0_lens0_no_data_error(sacc_galaxy_xis_lens0_lens0_no_data):
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0_no_data

    src0 = NumberCounts(sacc_tracer="lens0")

    statistic = TwoPoint("galaxy_density_xi", src0, src0)
    with pytest.warns(UserWarning, match="Empty index selected"):
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "for data type 'galaxy_density_xi' have no 2pt data in the SACC file "
                "and no input theta values were given"
            ),
        ):
            statistic.read(sacc_data)


def test_two_point_src0_src0_data_and_conf_warn(sacc_galaxy_cells_src0_src0_window):
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_window

    src0 = WeakLensing(sacc_tracer="src0")

    ell_config: EllOrThetaConfig = {
        "minimum": 1,
        "maximum": 100,
        "n": 5,
        "binning": "lin",
    }

    statistic = TwoPoint("galaxy_shear_cl_ee", src0, src0, ell_or_theta=ell_config)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "have 2pt data and you have specified `ell` in the configuration. "
            "`ell` is being ignored!"
        ),
    ):
        statistic.read(sacc_data)


def test_two_point_lens0_lens0_data_and_conf_warn(sacc_galaxy_xis_lens0_lens0):
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0

    src0 = NumberCounts(sacc_tracer="lens0")

    theta_config: EllOrThetaConfig = {
        "minimum": 0.0,
        "maximum": 1.0,
        "n": 5,
        "binning": "lin",
    }

    statistic = TwoPoint("galaxy_density_xi", src0, src0, ell_or_theta=theta_config)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "have 2pt data and you have specified `theta` in the configuration. "
            "`theta` is being ignored!"
        ),
    ):
        statistic.read(sacc_data)


def test_use_source_factory(harmonic_bin_1: InferredGalaxyZDist):
    wl_factory = WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    nc_factory = NumberCountsFactory(per_bin_systematics=[], global_systematics=[])

    measurement = list(harmonic_bin_1.measurements)[0]
    source = use_source_factory(harmonic_bin_1, measurement, wl_factory, nc_factory)

    if measurement in GALAXY_LENS_TYPES:
        assert isinstance(source, NumberCounts)
    elif measurement in GALAXY_SOURCE_TYPES:
        assert isinstance(source, WeakLensing)
    else:
        assert False, f"Unknown measurement type: {measurement}"


def test_use_source_factory_invalid_measurement(harmonic_bin_1: InferredGalaxyZDist):
    with pytest.raises(
        ValueError,
        match="Measurement .* not found in inferred galaxy redshift distribution .*",
    ):
        use_source_factory(harmonic_bin_1, Galaxies.SHEAR_MINUS, None, None)


def test_use_source_factory_metadata_only_counts():
    wl_factory = WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    nc_factory = NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
    source = use_source_factory_metadata_index(
        "bin1", Galaxies.COUNTS, wl_factory=wl_factory, nc_factory=nc_factory
    )
    assert isinstance(source, NumberCounts)


def test_use_source_factory_metadata_only_shear():
    wl_factory = WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    nc_factory = NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
    source = use_source_factory_metadata_index(
        "bin1", Galaxies.SHEAR_E, wl_factory=wl_factory, nc_factory=nc_factory
    )
    assert isinstance(source, WeakLensing)


def test_use_source_factory_metadata_only_invalid_measurement():
    wl_factory = WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    nc_factory = NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
    with pytest.raises(ValueError, match="Unknown measurement type encountered .*"):
        use_source_factory_metadata_index(
            "bin1", 120, wl_factory=wl_factory, nc_factory=nc_factory  # type: ignore
        )


def test_two_point_wrong_type():
    with pytest.raises(ValueError, match="The SACC data type cow is not supported!"):
        TwoPoint(
            "cow", WeakLensing(sacc_tracer="calma"), WeakLensing(sacc_tracer="fernando")
        )


def test_from_metadata_harmonic_wrong_metadata():
    with pytest.raises(
        ValueError, match=re.escape("Metadata of type <class 'str'> is not supported")
    ):
        TwoPoint._from_metadata_single(  # pylint: disable=protected-access
            metadata="NotAMetadata",  # type: ignore
        )


def test_use_source_factory_metadata_only_wrong_measurement():
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Measurement .* not supported!"):
        use_source_factory_metadata_index(
            "bin1", unknown_type, wl_factory=None, nc_factory=None
        )


def test_from_metadata_only_harmonic():
    wl_factory = WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    nc_factory = NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
    metadata: TwoPointHarmonicIndex = {
        "data_type": "galaxy_density_xi",
        "tracer_names": TracerNames("lens0", "lens0"),
    }
    two_point = TwoPoint.from_metadata_index(
        [metadata],
        wl_factory=wl_factory,
        nc_factory=nc_factory,
    ).pop()
    assert isinstance(two_point, TwoPoint)
    assert not two_point.ready


def test_from_metadata_only_real():
    wl_factory = WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    nc_factory = NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
    metadata: TwoPointRealIndex = {
        "data_type": "galaxy_shear_xi_plus",
        "tracer_names": TracerNames("src0", "src0"),
    }
    two_point = TwoPoint.from_metadata_index(
        [metadata],
        wl_factory=wl_factory,
        nc_factory=nc_factory,
    ).pop()
    assert isinstance(two_point, TwoPoint)
    assert not two_point.ready


def test_from_measurement_compute_theory_vector_window(
    harmonic_data_with_window: TwoPointMeasurement,
):
    wl_factory = WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    nc_factory = NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
    two_points = TwoPoint.from_measurement(
        [harmonic_data_with_window], wl_factory=wl_factory, nc_factory=nc_factory
    )
    two_point: TwoPoint = two_points.pop()

    assert isinstance(two_point, TwoPoint)
    assert two_point.ready

    req_params = two_point.required_parameters()
    default_values = req_params.get_default_values()
    params = ParamsMap(default_values)

    tools = ModelingTools()
    tools.update(params)
    tools.prepare(pyccl.CosmologyVanillaLCDM())
    two_point.update(params)

    prediction = two_point.compute_theory_vector(tools)

    assert isinstance(prediction, TheoryVector)
    assert prediction.shape == (4,)


def test_from_measurement_compute_theory_vector_window_check(
    harmonic_data_with_window: TwoPointMeasurement,
    harmonic_data_no_window: TwoPointMeasurement,
):
    wl_factory = WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
    nc_factory = NumberCountsFactory(per_bin_systematics=[], global_systematics=[])

    two_points_with_window = TwoPoint.from_measurement(
        [harmonic_data_with_window], wl_factory=wl_factory, nc_factory=nc_factory
    )
    two_point_with_window: TwoPoint = two_points_with_window.pop()

    two_points_without_window = TwoPoint.from_measurement(
        [harmonic_data_no_window], wl_factory=wl_factory, nc_factory=nc_factory
    )
    two_point_without_window: TwoPoint = two_points_without_window.pop()

    assert isinstance(two_point_with_window, TwoPoint)
    assert two_point_with_window.ready

    assert isinstance(two_point_without_window, TwoPoint)
    assert two_point_without_window.ready

    req_params = two_point_with_window.required_parameters()
    default_values = req_params.get_default_values()
    params = ParamsMap(default_values)

    tools = ModelingTools()
    tools.update(params)
    tools.prepare(pyccl.CosmologyVanillaLCDM())

    two_point_with_window.update(params)
    two_point_without_window.update(params)

    prediction_with_window = two_point_with_window.compute_theory_vector(tools)
    prediction_without_window = two_point_without_window.compute_theory_vector(tools)

    assert isinstance(prediction_with_window, TheoryVector)
    assert prediction_with_window.shape == (4,)

    assert isinstance(prediction_without_window, TheoryVector)
    assert prediction_without_window.shape == (100,)
    # Currently the C0 and C1 are set to 0 when a window is present, so we need to do
    # the same here.
    prediction_without_window[0:2] = 0.0

    binned_after = [
        np.mean(prediction_without_window[0:25]) * 1.0,
        np.mean(prediction_without_window[25:50]) * 2.0,
        np.mean(prediction_without_window[50:75]) * 3.0,
        np.mean(prediction_without_window[75:100]) * 4.0,
    ]
    assert_allclose(prediction_with_window, binned_after)
