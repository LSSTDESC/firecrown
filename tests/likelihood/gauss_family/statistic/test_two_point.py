"""
Tests for the TwoPoint module.
"""

import re
from unittest.mock import MagicMock
import numpy as np
from numpy.testing import assert_allclose
import pytest

from pydantic import BaseModel

from firecrown.updatable import get_default_params_map
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap

from firecrown.likelihood.number_counts import (
    NumberCounts,
)
from firecrown.likelihood.weak_lensing import (
    WeakLensing,
)
import firecrown.metadata_types as mdt
import firecrown.likelihood.two_point as tp
from firecrown.likelihood.two_point import (
    NumberCountsFactory,
    TwoPointFactory,
    use_source_factory,
    use_source_factory_metadata_index,
)
from firecrown.models.two_point import TwoPointTheory
from firecrown.generators.two_point import (
    generate_bin_centers,
    EllOrThetaConfig,
    LogLinearElls,
)
from firecrown.metadata_types import (
    Galaxies,
    InferredGalaxyZDist,
    GALAXY_LENS_TYPES,
    GALAXY_SOURCE_TYPES,
)
from firecrown.metadata_functions import (
    TwoPointHarmonicIndex,
    TwoPointRealIndex,
)
from firecrown.data_types import TwoPointMeasurement, TheoryVector


@pytest.fixture(name="include_rsd", params=[True, False], ids=["rsd", "no_rsd"])
def fixture_include_rsd(request) -> bool:
    """Return whether to include RSD in the test."""
    return request.param


@pytest.fixture(name="source_0")
def fixture_source_0() -> NumberCounts:
    """Return an almost-default NumberCounts source."""
    return NumberCounts(sacc_tracer="lens_0")


@pytest.fixture(name="tools")
def fixture_tools() -> ModelingTools:
    """Return a trivial ModelingTools object."""
    return ModelingTools()


@pytest.fixture(name="two_point_with_window")
def fixture_two_point_with_window(
    harmonic_data_with_window: TwoPointMeasurement, tp_factory: TwoPointFactory
) -> tp.TwoPoint:
    """Return a TwoPoint object with a window."""
    two_points = tp.TwoPoint.from_measurement(
        [harmonic_data_with_window], tp_factory=tp_factory
    )
    return two_points.pop()


@pytest.fixture(name="two_point_without_window")
def fixture_two_point_without_window(
    harmonic_data_no_window: TwoPointMeasurement, tp_factory: TwoPointFactory
) -> tp.TwoPoint:
    """Return a TwoPoint object without a window."""
    two_points = tp.TwoPoint.from_measurement(
        [harmonic_data_no_window], tp_factory=tp_factory
    )
    return two_points.pop()


@pytest.fixture(name="apply_interp_when", params=list(tp.ApplyInterpolationWhen))
def fixture_apply_interp_when(request) -> tp.ApplyInterpolationWhen:
    return request.param


def test_log_linear_ells_no_rounding() -> None:
    res = LogLinearElls(minimum=0, midpoint=5, maximum=80, n_log=5).generate()
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    assert res.shape == expected.shape
    assert np.allclose(expected, res)


def test_log_linear_ells_doing_rounding() -> None:
    res = LogLinearElls(minimum=1, midpoint=3, maximum=100, n_log=5).generate()
    expected = np.array([1.0, 2.0, 3.0, 7.0, 17.0, 42.0, 100.0])
    assert np.allclose(expected, res)


def test_log_linear_ells_generate_all() -> None:
    res = LogLinearElls(minimum=1, midpoint=3, maximum=100, n_log=5).generate_all()
    expected = np.arange(1, 100 + 1)
    assert np.allclose(expected, res)


def test_compute_theory_vector(source_0: NumberCounts) -> None:
    # To create the TwoPoint object we need at least one source.
    statistic = tp.TwoPoint("galaxy_density_xi", source_0, source_0)
    assert isinstance(statistic, tp.TwoPoint)

    # Before calling compute_theory_vector, we must get the TwoPoint object
    # into the correct state.
    # prediction = statistic.compute_theory_vector(tools)
    # assert isinstance(prediction, TheoryVector)


def test_tracer_names() -> None:
    assert tp.TracerNames("", "") == mdt.TRACER_NAMES_TOTAL

    tn1 = tp.TracerNames("cow", "pig")
    assert tn1[0] == "cow"
    assert tn1[1] == "pig"

    tn2 = tp.TracerNames("cat", "dog")
    assert tn1 != tn2
    assert hash(tn1) != hash(tn2)

    with pytest.raises(IndexError):
        _ = tn1[2]


def test_two_point_src0_src0_window(sacc_galaxy_cells_src0_src0_window) -> None:
    """This test also makes sure that TwoPoint theory calculations are
    repeatable."""
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_window

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = tp.TwoPoint("galaxy_shear_cl_ee", src0, src0)
    statistic.read(sacc_data)

    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    assert statistic.window is not None

    statistic.reset()
    statistic.update(params)
    tools.update(params)
    result1 = statistic.compute_theory_vector(tools)
    assert all(np.isfinite(result1))

    statistic.reset()
    statistic.update(params)
    tools.update(params)
    result2 = statistic.compute_theory_vector(tools)
    assert np.array_equal(result1, result2)


def test_two_point_src0_src0_no_window(sacc_galaxy_cells_src0_src0_no_window) -> None:
    """This test also makes sure that TwoPoint theory calculations are
    repeatable."""
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_window

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = tp.TwoPoint("galaxy_shear_cl_ee", src0, src0)
    statistic.read(sacc_data)

    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    assert statistic.window is None

    statistic.reset()
    statistic.update(params)
    tools.update(params)
    result1 = statistic.compute_theory_vector(tools)
    assert all(np.isfinite(result1))

    statistic.reset()
    statistic.update(params)
    tools.update(params)
    result2 = statistic.compute_theory_vector(tools)
    assert np.array_equal(result1, result2)


def test_two_point_generate_ell_or_theta() -> None:
    # Logarithmic binning.
    assert np.allclose(
        generate_bin_centers(minimum=1.0, maximum=1000.0, n=3, binning="log"),
        np.array(np.sqrt([10.0, 1000.0, 100000.0])),
    )

    # Linear binning.
    assert np.array_equal(
        generate_bin_centers(minimum=0.0, maximum=12.0, n=6, binning="lin"),
        np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]),
    )

    # Invalid binning.
    with pytest.raises(ValueError, match="Unrecognized binning: cow"):
        generate_bin_centers(minimum=1, maximum=100, n=5, binning="cow")


def test_two_point_src0_src0_no_data_lin(sacc_galaxy_cells_src0_src0_no_data) -> None:
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_data

    src0 = WeakLensing(sacc_tracer="src0")

    ell_config: EllOrThetaConfig = {
        "minimum": 1,
        "maximum": 100,
        "n": 5,
        "binning": "lin",
    }

    statistic = tp.TwoPoint(
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


def test_two_point_src0_src0_no_data_log(sacc_galaxy_cells_src0_src0_no_data) -> None:
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_data

    src0 = WeakLensing(sacc_tracer="src0")

    ell_config: EllOrThetaConfig = {
        "minimum": 1,
        "maximum": 100,
        "n": 5,
        "binning": "log",
    }

    statistic = tp.TwoPoint(
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


def test_two_point_lens0_lens0_no_data(sacc_galaxy_xis_lens0_lens0_no_data) -> None:
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0_no_data

    src0 = NumberCounts(sacc_tracer="lens0")

    ell_config: EllOrThetaConfig = {
        "minimum": 0.0,
        "maximum": 1.0,
        "n": 5,
        "binning": "lin",
    }

    statistic = tp.TwoPoint(
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


def test_two_point_theory_construction() -> None:
    src0 = WeakLensing(sacc_tracer="src0")
    theory = TwoPointTheory(
        sacc_data_type="galaxy_shear_cl_ee",
        sources=(src0, src0),
    )
    assert theory.sacc_data_type == "galaxy_shear_cl_ee"
    assert theory.source0 is src0
    assert theory.source1 is src0


def test_two_point_lens0_lens0_config(sacc_galaxy_xis_lens0_lens0) -> None:
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0

    src0 = NumberCounts(sacc_tracer="lens0")

    statistic = tp.TwoPoint(
        "galaxy_density_xi",
        src0,
        src0,
        interp_ells_gen=LogLinearElls(minimum=2, midpoint=6, n_log=180),
    )
    statistic.read(sacc_data)

    tools = ModelingTools()
    params = get_default_params_map(tools)
    params.update({"lens0_bias": 1.0})
    tools.update(params)
    tools.prepare()

    assert statistic.window is None
    assert statistic.thetas is not None
    assert all(np.isfinite(statistic.thetas))
    assert statistic.ells_for_xi is not None
    assert all(np.isfinite(statistic.ells_for_xi))
    # The size depends on the configuration but also
    # on how many unique ells we get from the log-binning.
    assert len(statistic.ells_for_xi) == 175

    statistic.update(params)
    statistic.compute_theory_vector(tools)


def test_two_point_src0_src0_no_data_error(sacc_galaxy_cells_src0_src0_no_data) -> None:
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_data

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = tp.TwoPoint("galaxy_shear_cl_ee", src0, src0)
    with pytest.warns(UserWarning, match="Empty index selected"):
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "for data type 'galaxy_shear_cl_ee' have no 2pt data in the SACC file "
                "and no input ell values were given"
            ),
        ):
            statistic.read(sacc_data)


def test_two_point_lens0_lens0_no_data_error(
    sacc_galaxy_xis_lens0_lens0_no_data,
) -> None:
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0_no_data

    src0 = NumberCounts(sacc_tracer="lens0")

    statistic = tp.TwoPoint("galaxy_density_xi", src0, src0)
    with pytest.warns(UserWarning, match="Empty index selected"):
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "for data type 'galaxy_density_xi' have no 2pt data in the SACC file "
                "and no input theta values were given"
            ),
        ):
            statistic.read(sacc_data)


def test_two_point_src0_src0_data_and_conf_warn(
    sacc_galaxy_cells_src0_src0_window,
) -> None:
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_window

    src0 = WeakLensing(sacc_tracer="src0")

    ell_config: EllOrThetaConfig = {
        "minimum": 1,
        "maximum": 100,
        "n": 5,
        "binning": "lin",
    }

    statistic = tp.TwoPoint("galaxy_shear_cl_ee", src0, src0, ell_or_theta=ell_config)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "have 2pt data and you have specified `ell` in the configuration. "
            "`ell` is being ignored!"
        ),
    ):
        statistic.read(sacc_data)


def test_two_point_lens0_lens0_data_and_conf_warn(sacc_galaxy_xis_lens0_lens0) -> None:
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0

    src0 = NumberCounts(sacc_tracer="lens0")

    theta_config: EllOrThetaConfig = {
        "minimum": 0.0,
        "maximum": 1.0,
        "n": 5,
        "binning": "lin",
    }

    statistic = tp.TwoPoint("galaxy_density_xi", src0, src0, ell_or_theta=theta_config)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "have 2pt data and you have specified `theta` in the configuration. "
            "`theta` is being ignored!"
        ),
    ):
        statistic.read(sacc_data)


def test_use_source_factory(
    harmonic_bin_1: InferredGalaxyZDist, tp_factory: TwoPointFactory
) -> None:
    measurement = list(harmonic_bin_1.measurements)[0]
    source = use_source_factory(harmonic_bin_1, measurement, tp_factory)

    factory = tp_factory.get_factory(measurement, harmonic_bin_1.type_source)
    if measurement in GALAXY_LENS_TYPES:
        assert isinstance(source, NumberCounts)
        assert isinstance(factory, NumberCountsFactory)
        assert source.has_rsd == factory.include_rsd
    elif measurement in GALAXY_SOURCE_TYPES:
        assert isinstance(source, WeakLensing)
    else:
        assert False, f"Unknown measurement type: {measurement}"


def test_use_source_factory_invalid_measurement(
    harmonic_bin_1: InferredGalaxyZDist, tp_factory: TwoPointFactory
) -> None:
    with pytest.raises(
        ValueError,
        match="Measurement .* not found in inferred galaxy redshift distribution .*",
    ):
        use_source_factory(harmonic_bin_1, Galaxies.PART_OF_XI_MINUS, tp_factory)


def test_use_source_factory_metadata_only_counts(tp_factory: TwoPointFactory) -> None:
    source = use_source_factory_metadata_index(
        "bin1", Galaxies.COUNTS, tp_factory=tp_factory
    )
    factory = tp_factory.get_factory(Galaxies.COUNTS)
    assert isinstance(factory, NumberCountsFactory)
    assert isinstance(source, NumberCounts)
    assert source.has_rsd == factory.include_rsd


def test_use_source_factory_metadata_only_shear(tp_factory: TwoPointFactory) -> None:
    source = use_source_factory_metadata_index(
        "bin1", Galaxies.SHEAR_E, tp_factory=tp_factory
    )
    factory = tp_factory.get_factory(Galaxies.SHEAR_E)
    assert isinstance(factory, tp.WeakLensingFactory)
    assert isinstance(source, WeakLensing)


def test_use_source_factory_metadata_only_invalid_measurement(
    tp_factory: TwoPointFactory,
) -> None:
    with pytest.raises(ValueError, match="Unknown measurement type encountered .*"):
        use_source_factory_metadata_index("bin1", 120, tp_factory)  # type: ignore


def test_two_point_wrong_type() -> None:
    with pytest.raises(ValueError, match="The SACC data type cow is not supported!"):
        tp.TwoPoint(
            "cow", WeakLensing(sacc_tracer="calma"), WeakLensing(sacc_tracer="fernando")
        )


def test_from_metadata_harmonic_wrong_metadata(tp_factory: TwoPointFactory) -> None:
    with pytest.raises(
        ValueError, match=re.escape("Metadata of type <class 'str'> is not supported")
    ):
        tp.TwoPoint._from_metadata_single(  # pylint: disable=protected-access
            metadata="NotAMetadata", tp_factory=tp_factory  # type: ignore
        )


def test_use_source_factory_metadata_only_wrong_measurement(
    tp_factory: TwoPointFactory,
) -> None:
    unknown_type = MagicMock()
    unknown_type.configure_mock(__eq__=MagicMock(return_value=False))

    with pytest.raises(ValueError, match="Unknown measurement type encountered"):
        use_source_factory_metadata_index("bin1", unknown_type, tp_factory)


def test_from_metadata_only_harmonic(tp_factory: TwoPointFactory) -> None:
    metadata: TwoPointHarmonicIndex = {
        "data_type": "galaxy_density_xi",
        "tracer_names": tp.TracerNames("lens0", "lens0"),
    }
    two_point = tp.TwoPoint.from_metadata_index([metadata], tp_factory).pop()
    assert isinstance(two_point, tp.TwoPoint)
    assert not two_point.ready


def test_from_metadata_only_real(tp_factory: TwoPointFactory) -> None:
    metadata: TwoPointRealIndex = {
        "data_type": "galaxy_shear_xi_plus",
        "tracer_names": tp.TracerNames("src0", "src0"),
    }
    two_point = tp.TwoPoint.from_metadata_index([metadata], tp_factory).pop()
    assert isinstance(two_point, tp.TwoPoint)
    assert not two_point.ready


def test_from_measurement_compute_theory_vector_window(
    two_point_with_window: tp.TwoPoint,
) -> None:
    assert isinstance(two_point_with_window, tp.TwoPoint)
    assert two_point_with_window.ready

    tools = ModelingTools()
    req_params = (
        two_point_with_window.required_parameters() + tools.required_parameters()
    )
    default_values = req_params.get_default_values()
    params = ParamsMap(default_values)

    tools.update(params)
    tools.prepare()
    two_point_with_window.update(params)

    prediction = two_point_with_window.compute_theory_vector(tools)

    assert isinstance(prediction, TheoryVector)
    assert prediction.shape == (4,)


def test_from_measurement_compute_theory_vector_window_check(
    two_point_with_window: tp.TwoPoint, two_point_without_window: tp.TwoPoint
) -> None:
    assert isinstance(two_point_with_window, tp.TwoPoint)
    assert two_point_with_window.ready

    assert isinstance(two_point_without_window, tp.TwoPoint)
    assert two_point_without_window.ready

    tools = ModelingTools()
    req_params = (
        two_point_with_window.required_parameters() + tools.required_parameters()
    )
    default_values = req_params.get_default_values()
    params = ParamsMap(default_values)

    tools.update(params)
    tools.prepare()
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


def test_apply_interp_when_none() -> None:
    apply_interp_when = tp.ApplyInterpolationWhen.NONE
    assert apply_interp_when == tp.ApplyInterpolationWhen.NONE

    for app_when in tp.ApplyInterpolationWhen:
        assert not (app_when & apply_interp_when)


def test_apply_interp_when_all() -> None:
    apply_interp_when = tp.ApplyInterpolationWhen.ALL
    assert apply_interp_when == tp.ApplyInterpolationWhen.ALL

    for app_when in tp.ApplyInterpolationWhen:
        assert app_when & apply_interp_when


def test_apply_interp_when_default() -> None:
    apply_interp_when = tp.ApplyInterpolationWhen.DEFAULT
    assert apply_interp_when == tp.ApplyInterpolationWhen.DEFAULT

    assert apply_interp_when & tp.ApplyInterpolationWhen.REAL
    assert apply_interp_when & tp.ApplyInterpolationWhen.HARMONIC_WINDOW
    assert not (apply_interp_when & tp.ApplyInterpolationWhen.HARMONIC)


class ModelWithApplyInterpWhen(BaseModel):
    """ApplyInterpolationWhen test class."""

    apply_interp_when: tp.ApplyInterpolationWhen


def test_apply_interp_when_serialization_default():
    model = ModelWithApplyInterpWhen(
        apply_interp_when=tp.ApplyInterpolationWhen.DEFAULT
    )
    assert model.apply_interp_when == tp.ApplyInterpolationWhen.DEFAULT

    model_dict = model.model_dump()
    assert model_dict["apply_interp_when"] == "REAL|HARMONIC_WINDOW"

    model = ModelWithApplyInterpWhen.model_validate(model_dict)
    assert model.apply_interp_when == tp.ApplyInterpolationWhen.DEFAULT


def test_apply_interp_when_serialization_all():
    model = ModelWithApplyInterpWhen(apply_interp_when=tp.ApplyInterpolationWhen.ALL)
    assert model.apply_interp_when == tp.ApplyInterpolationWhen.ALL

    model_dict = model.model_dump()
    assert model_dict["apply_interp_when"] == "REAL|HARMONIC|HARMONIC_WINDOW"

    model = ModelWithApplyInterpWhen.model_validate(model_dict)
    assert model.apply_interp_when == tp.ApplyInterpolationWhen.ALL


def test_apply_interp_when_serialization_none():
    model = ModelWithApplyInterpWhen(apply_interp_when=tp.ApplyInterpolationWhen.NONE)
    assert model.apply_interp_when == tp.ApplyInterpolationWhen.NONE

    model_dict = model.model_dump()
    assert model_dict["apply_interp_when"] == "NONE"

    model = ModelWithApplyInterpWhen.model_validate(model_dict)
    assert model.apply_interp_when == tp.ApplyInterpolationWhen.NONE


def test_apply_interp_when_serialization(apply_interp_when: tp.ApplyInterpolationWhen):
    model = ModelWithApplyInterpWhen(apply_interp_when=apply_interp_when)
    assert model.apply_interp_when == apply_interp_when

    model_dict = model.model_dump()
    assert model_dict["apply_interp_when"] == apply_interp_when.name

    model = ModelWithApplyInterpWhen.model_validate(model_dict)
    assert model.apply_interp_when == apply_interp_when


def test_apply_interp_when_serialization_error():
    with pytest.raises(
        TypeError, match="Cannot parse ApplyInterpolationWhen from value: None"
    ):
        ModelWithApplyInterpWhen.model_validate({"apply_interp_when": None})

    with pytest.raises(ValueError, match="Value error, Invalid flag name"):
        ModelWithApplyInterpWhen.model_validate({"apply_interp_when": ""})

    with pytest.raises(ValueError, match="Value error, Invalid flag name"):
        ModelWithApplyInterpWhen.model_validate({"apply_interp_when": "foo"})


def test_two_point_src0_src0_window_aiw(
    sacc_galaxy_cells_src0_src0_window: tuple,
    apply_interp_when: tp.ApplyInterpolationWhen,
) -> None:
    """This test also makes sure that TwoPoint theory calculations are
    repeatable."""
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_window

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = tp.TwoPoint(
        "galaxy_shear_cl_ee", src0, src0, apply_interp=apply_interp_when
    )
    statistic.read(sacc_data)

    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    statistic.reset()
    statistic.update(params)
    tools.update(params)

    result1 = statistic.compute_theory_vector(tools)
    assert all(np.isfinite(result1))
    cells = statistic.theory.cells[mdt.TRACER_NAMES_TOTAL]
    assert statistic.theory.ells is not None
    if apply_interp_when & tp.ApplyInterpolationWhen.HARMONIC_WINDOW:
        assert len(cells) != len(statistic.theory.ells)
    else:
        assert len(cells) == len(statistic.theory.ells)


def test_two_point_src0_src0_no_window_aiw(
    sacc_galaxy_cells_src0_src0_no_window: tuple,
    apply_interp_when: tp.ApplyInterpolationWhen,
) -> None:
    sacc_data, _, _ = sacc_galaxy_cells_src0_src0_no_window

    src0 = WeakLensing(sacc_tracer="src0")

    statistic = tp.TwoPoint(
        "galaxy_shear_cl_ee", src0, src0, apply_interp=apply_interp_when
    )
    statistic.read(sacc_data)

    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    assert statistic.window is None

    statistic.reset()
    statistic.update(params)
    tools.update(params)
    result1 = statistic.compute_theory_vector(tools)
    assert all(np.isfinite(result1))

    cells = statistic.theory.cells[mdt.TRACER_NAMES_TOTAL]
    assert statistic.theory.ells is not None
    if apply_interp_when & tp.ApplyInterpolationWhen.HARMONIC:
        assert len(cells) != len(statistic.theory.ells)
    else:
        assert len(cells) == len(statistic.theory.ells)


def test_two_point_src0_src0_real_aiw(
    sacc_galaxy_xis_lens0_lens0_real: tuple,
    apply_interp_when: tp.ApplyInterpolationWhen,
) -> None:
    sacc_data, _, _ = sacc_galaxy_xis_lens0_lens0_real

    lens0 = WeakLensing(sacc_tracer="lens0")
    statistic = tp.TwoPoint(
        "galaxy_density_xi", lens0, lens0, apply_interp=apply_interp_when
    )
    statistic.read(sacc_data)

    tools = ModelingTools()
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

    statistic.reset()
    statistic.update(params)
    tools.update(params)
    result1 = statistic.compute_theory_vector(tools)
    assert all(np.isfinite(result1))
    cells = statistic.theory.cells[mdt.TRACER_NAMES_TOTAL]
    if apply_interp_when & tp.ApplyInterpolationWhen.REAL:
        assert len(cells) == len(statistic.theory.ells_for_xi)
    else:
        assert len(cells) != len(statistic.theory.ells_for_xi)
