"""Integer domain storage at public metadata and measurement boundaries."""

import numpy as np
import numpy.typing as npt
import pytest
import sacc

from firecrown.data_functions import extract_all_harmonic_data
from firecrown.metadata_functions import extract_all_harmonic_metadata
from firecrown.likelihood import TwoPoint
from firecrown.likelihood.number_counts import NumberCounts
from firecrown.modeling_tools import ModelingTools
from firecrown.models.two_point import ApplyInterpolationWhen
from firecrown.updatable import get_default_params_map

from firecrown.data_types import TwoPointMeasurement
from firecrown.metadata_types import TwoPointHarmonic, TwoPointXY


@pytest.mark.parametrize(
    "ells",
    [
        np.array([2, 5, 11], dtype=np.int32),
        np.array([2, 5, 11], dtype=np.uint64),
        np.array([2, 5, 11], dtype=np.int64),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_harmonic_integer_storage(
    harmonic_two_point_xy: TwoPointXY,
    ells: npt.NDArray[np.integer[npt.NBitBase]],
    empty: bool,
) -> None:
    """Convert supported integer representations without changing multipoles.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param ells: Input integer representation.
    :param empty: Exercise an empty exact-multipole array.
    """
    values = [] if empty else [2, 5, 11]
    ells = ells[:0] if empty else ells
    metadata = TwoPointHarmonic(XY=harmonic_two_point_xy, ells=ells)
    assert metadata.ells.dtype == np.int64
    np.testing.assert_array_equal(metadata.ells, values)
    if ells.dtype == np.int64 and not empty:
        assert np.shares_memory(metadata.ells, ells)


@pytest.mark.parametrize(
    "ells",
    [
        np.array([2.0, 5.0]),
        np.array([2.5, 5.5]),
        np.array([True, False]),
        np.array([2 + 0j, 5 + 0j]),
        np.array(["2", "5"]),
        np.array([2, 5], dtype=object),
        np.zeros(2, dtype=[("ell", np.int64)]),
        np.zeros(2, dtype="datetime64[D]"),
        np.ma.array([2, 5], mask=False),
        np.ma.array([2, 5], mask=[False, True]),
    ],
)
def test_harmonic_rejects_noninteger_storage(
    harmonic_two_point_xy: TwoPointXY, ells: npt.NDArray[np.generic]
) -> None:
    """Reject noninteger or masked window support before value conversion.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param ells: Deliberately unsupported exact-multipole representation.
    """
    with pytest.raises(TypeError, match="integer dtype|masked"):
        TwoPointHarmonic(
            XY=harmonic_two_point_xy,
            ells=ells,  # type: ignore[arg-type]  # Invalid runtime ingress.
            window=np.ones((2, 1)),
            window_ells=np.array([3.5]),
        )


def test_harmonic_rejects_uint64_overflow(harmonic_two_point_xy: TwoPointXY) -> None:
    """Reject unsigned values outside int64 before narrowing.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    """
    with pytest.raises(ValueError, match="int64 range"):
        TwoPointHarmonic(
            XY=harmonic_two_point_xy,
            ells=np.array([np.iinfo(np.int64).max + 1], dtype=np.uint64),
        )


@pytest.mark.parametrize(
    "indices",
    [
        np.array([2, 0], dtype=np.int32),
        np.array([2, 0], dtype=np.uint64),
        np.array([2, 0], dtype=np.int64),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_measurement_integer_indices(
    harmonic_two_point_xy: TwoPointXY,
    indices: npt.NDArray[np.integer[npt.NBitBase]],
    empty: bool,
) -> None:
    """Store fixed-width indices usable in actual indexing, including empties.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param indices: Input integer representation.
    :param empty: Exercise an empty selection.
    """
    indices = indices[:0] if empty else indices
    measurement = TwoPointMeasurement(
        data=np.array([] if empty else [30.0, 10.0]),
        indices=indices,
        covariance_name="cov",
        metadata=TwoPointHarmonic(
            XY=harmonic_two_point_xy,
            ells=np.array([] if empty else [2, 5], dtype=np.int64),
        ),
    )
    assert measurement.indices.dtype == np.int64
    np.testing.assert_array_equal(
        np.array([10.0, 20.0, 30.0])[measurement.indices], measurement.data
    )


@pytest.mark.parametrize(
    "indices",
    [
        np.array([0.0, 1.0]),
        np.array([0.5, 1.5]),
        np.array([False, True]),
        np.ma.array([0, 1], mask=False),
        np.array([0 + 0j, 1 + 0j]),
        np.array(["0", "1"]),
        np.array([0, 1], dtype=object),
        np.zeros(2, dtype=[("index", np.int64)]),
        np.zeros(2, dtype="datetime64[D]"),
    ],
)
def test_measurement_rejects_noninteger_indices(
    harmonic_two_point_xy: TwoPointXY, indices: npt.NDArray[np.generic]
) -> None:
    """Reject fractional and integral-valued floating indices before selection.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param indices: Invalid index representation.
    """
    with pytest.raises(TypeError, match="integer dtype|masked"):
        TwoPointMeasurement(
            data=np.array([1.0, 2.0]),
            indices=indices,  # type: ignore[arg-type]  # Invalid runtime ingress.
            covariance_name="cov",
            metadata=TwoPointHarmonic(
                XY=harmonic_two_point_xy, ells=np.array([2, 5], dtype=np.int64)
            ),
        )


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("floating_support", [False, True])
def test_sacc_support_and_fractional_labels(
    windowed: bool, floating_support: bool
) -> None:
    """Distinguish exact SACC support from fractional observation labels.

    :param windowed: Use fractional labels on a bandpower window.
    :param floating_support: Supply integral-valued floating exact support.
    """
    data = sacc.Sacc()
    data.add_tracer("NZ", "lens0", z=np.array([0.1, 0.5, 1.0]), nz=np.ones(3))
    support = np.array([2, 5, 11], dtype=np.int32)
    supplied_support = support.astype(np.float64) if floating_support else support
    window = (
        sacc.BandpowerWindow(
            supplied_support, np.array([[0.5, 0.0], [0.5, 0.0], [0.0, 1.0]])
        )
        if windowed
        else None
    )
    if windowed:
        data.add_ell_cl(
            "galaxy_density_cl",
            "lens0",
            "lens0",
            np.array([2.5, 7.5]),
            np.array([1.0, 2.0]),
            window=window,
        )
    else:
        # SACC's add_ell_cl explicitly casts scalar tags to float. Construct
        # exact support with add_data_point to retain the supplied representation.
        for ell, value in zip(supplied_support, [1.0, 2.0, 3.0]):
            data.add_data_point("galaxy_density_cl", ("lens0", "lens0"), value, ell=ell)
    data.add_covariance(np.eye(2 if windowed else 3))
    source = NumberCounts(sacc_tracer="lens0")
    statistic = TwoPoint("galaxy_density_cl", source, source)
    if floating_support and windowed:
        for extract in (extract_all_harmonic_metadata, extract_all_harmonic_data):
            with pytest.raises(TypeError, match="integer dtype"):
                extract(data)
        with pytest.raises(TypeError, match="integer dtype"):
            statistic.read(data)
        return
    metadata = extract_all_harmonic_metadata(data)[0]
    measurement = extract_all_harmonic_data(data)[0]
    statistic.read(data)
    expected_dtype = np.float64 if floating_support else np.int64
    assert metadata.ells.dtype == expected_dtype
    np.testing.assert_array_equal(metadata.ells, [2, 5, 11])
    assert measurement.indices.dtype == np.int64
    np.testing.assert_array_equal(data.mean[measurement.indices], measurement.data)
    assert statistic.ells is not None and statistic.ells.dtype == expected_dtype
    assert statistic.sacc_indices is not None
    assert statistic.sacc_indices.dtype == np.int64
    if windowed:
        assert metadata.window_ells is not None
        assert metadata.window_ells.dtype == np.float64
        np.testing.assert_array_equal(metadata.window_ells, [2.5, 7.5])


@pytest.mark.parametrize(
    "apply_interp", [ApplyInterpolationWhen.NONE, ApplyInterpolationWhen.HARMONIC]
)
@pytest.mark.parametrize(
    "ells",
    [
        np.array([2.5, 7.5], dtype=np.float64),
        np.array([200.5, 300.5], dtype=np.float64),
    ],
)
def test_fractional_unwindowed_multipoles(
    apply_interp: ApplyInterpolationWhen, ells: npt.NDArray[np.float64]
) -> None:
    """Preserve real unwindowed multipoles through extraction and prediction.

    :param apply_interp: Exercise direct and interpolated angular-Cl sampling.
    :param ells: Sampling grid below or above the interpolation midpoint.
    """
    data = sacc.Sacc()
    z = np.linspace(0.1, 1.0, 100)
    data.add_tracer("NZ", "lens0", z=z, nz=np.ones_like(z))
    data.add_ell_cl("galaxy_density_cl", "lens0", "lens0", ells, np.array([1.0, 2.0]))
    data.add_covariance(np.eye(2))

    metadata = extract_all_harmonic_metadata(data)[0]
    measurement = extract_all_harmonic_data(data)[0]
    source = NumberCounts(sacc_tracer="lens0")
    statistic = TwoPoint("galaxy_density_cl", source, source, apply_interp=apply_interp)
    statistic.read(data)

    assert metadata.ells.dtype == np.float64
    np.testing.assert_array_equal(metadata.ells, ells)
    assert measurement.indices.dtype == np.int64
    np.testing.assert_array_equal(data.mean[measurement.indices], measurement.data)
    assert statistic.ells is not None and statistic.ells.dtype == np.float64
    np.testing.assert_array_equal(statistic.ells, ells)
    assert statistic.sacc_indices is not None
    assert statistic.sacc_indices.dtype == np.int64

    tools = ModelingTools()
    params = get_default_params_map(statistic, tools)
    statistic.update(params)
    tools.update(params)
    tools.prepare()
    prediction = statistic.compute_theory_vector(tools)
    assert prediction.shape == (2,)
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize("ells", [np.array([np.nan]), np.array([np.inf])])
def test_harmonic_rejects_nonfinite_sampling(
    harmonic_two_point_xy: TwoPointXY, ells: npt.NDArray[np.float64]
) -> None:
    """Reject unwindowed sampling coordinates that cannot be evaluated.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param ells: Nonfinite real sampling coordinate.
    """
    with pytest.raises(ValueError, match="must be finite"):
        TwoPointHarmonic(XY=harmonic_two_point_xy, ells=ells)


def test_measurement_rejects_index_overflow(harmonic_two_point_xy: TwoPointXY) -> None:
    """Do not wrap an unsigned covariance index into a negative int64 index.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    """
    with pytest.raises(ValueError, match="int64 range"):
        TwoPointMeasurement(
            data=np.array([1.0]),
            indices=np.array([np.iinfo(np.int64).max + 1], dtype=np.uint64),
            covariance_name="cov",
            metadata=TwoPointHarmonic(
                XY=harmonic_two_point_xy, ells=np.array([2], dtype=np.int64)
            ),
        )
