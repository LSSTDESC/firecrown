"""Float64 contracts for windows, measurements and real coordinates at ingress.

Expected values are written as literals or computed from independent float64
arithmetic; no production converter generates an expectation.
"""

import hashlib
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import sacc

from firecrown.data_functions import (
    TwoPointBinFilter,
    TwoPointBinFilterCollection,
    extract_all_harmonic_data,
    extract_all_real_data,
)
from firecrown.data_types import TwoPointMeasurement
from firecrown.metadata_functions import (
    extract_all_harmonic_metadata,
    extract_all_real_metadata,
    extract_window_function,
)
from firecrown.metadata_types import (
    Galaxies,
    TwoPointFilterMethod,
    TwoPointHarmonic,
    TwoPointReal,
    TwoPointXY,
)

_RealDtype = type[np.float64] | type[np.float32] | type[np.int32] | type[np.int64]
_RealArray = npt.NDArray[np.float64]


def _real(values: object, dtype: _RealDtype) -> _RealArray:
    """Build a real array at the given dtype for ingress tests.

    :param values: Literal array contents.
    :param dtype: Storage dtype to construct.
    :returns: The array, typed as float64 for the static ingress signature; its
        runtime dtype is the requested one.
    """
    return cast(_RealArray, np.array(values, dtype=dtype))


# Support multipoles 2..5, two bandpowers. Column sums are 16 and 12.
SUPPORT = np.array([2, 3, 4, 5], dtype=np.int64)
WEIGHTS = [[1, 0], [3, 2], [5, 4], [7, 6]]
COLUMN_SUMS = [16.0, 12.0]
LABELS = [3.5, 4.5]
DATA = [1, 2]

REAL_DTYPES = [np.float64, np.float32, np.int32, np.int64]


def _rejected_arrays(shape: tuple[int, ...]) -> list[npt.NDArray[np.generic]]:
    """Return arrays whose physical families are unsupported at real ingress.

    :param shape: Shape of each rejected array.
    :returns: Boolean, complex, string, object and masked arrays.
    """
    return [
        np.ones(shape, dtype=np.bool_),
        np.ones(shape, dtype=np.complex128),
        np.full(shape, "1"),
        np.ones(shape, dtype=object),
        np.ma.array(np.ones(shape), mask=False),
    ]


def _windowed_sacc(
    dtype: _RealDtype,
    weights: list[list[int]] = WEIGHTS,
) -> sacc.Sacc:
    """Build harmonic SACC data with a bandpower window of the given dtype.

    :param dtype: Storage dtype of the weights, bandpowers and covariance.
    :param weights: Literal weight matrix, one row per support multipole.
    :returns: SACC data with one window, real labels and a covariance.
    """
    sacc_data = sacc.Sacc()
    z = np.linspace(0.0, 1.0, 20)
    sacc_data.add_tracer("NZ", "src0", z, np.exp(-((z - 0.5) ** 2) / 0.02))
    window = sacc.BandpowerWindow(SUPPORT, _real(weights, dtype))
    sacc_data.add_ell_cl(
        "galaxy_shear_cl_ee",
        "src0",
        "src0",
        np.array(LABELS),
        np.array(DATA, dtype=dtype),
        window=window,
    )
    sacc_data.add_covariance(np.diag(np.array([4, 9], dtype=dtype)))
    return sacc_data


def _window_indices(sacc_data: sacc.Sacc) -> npt.NDArray[np.int64]:
    """Return the integer SACC indices of the single windowed spectrum.

    :param sacc_data: SACC data containing the spectrum.
    :returns: Indices of its data points.
    """
    _, _, indices = sacc_data.get_ell_cl(
        "galaxy_shear_cl_ee", "src0", "src0", return_cov=False, return_ind=True
    )
    return np.asarray(indices, dtype=np.int64)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("normalize", [True, False])
def test_extract_window_function_converts_weights(
    dtype: _RealDtype, normalize: bool
) -> None:
    """Return float64 weights, keeping support integer and the scale option separate.

    :param dtype: Weight dtype stored by SACC.
    :param normalize: Existing sum-to-one window rescaling option.
    """
    sacc_data = _windowed_sacc(dtype)
    ells, weights = extract_window_function(
        sacc_data, _window_indices(sacc_data), normalize=normalize
    )
    assert ells is not None and weights is not None
    assert ells.dtype == np.int64
    np.testing.assert_array_equal(ells, SUPPORT)
    assert weights.dtype == np.float64
    raw = np.array(WEIGHTS, dtype=np.float64)
    if normalize:
        np.testing.assert_array_equal(weights, raw / np.array(COLUMN_SUMS))
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, rtol=1e-15)
    else:
        # The supplied scale is preserved: dtype conversion never forces unit sums.
        np.testing.assert_array_equal(weights, raw)
        np.testing.assert_array_equal(weights.sum(axis=0), COLUMN_SUMS)


@pytest.mark.parametrize("normalize", [True, False])
def test_extract_window_function_accumulates_in_float64(normalize: bool) -> None:
    """Sum float32 weights in float64, not after a float32 accumulation.

    The column holds 2**24 + 3 in float64. A float32 accumulation sticks at 2**24,
    so only conversion before the sum reproduces the independent total.

    :param normalize: Existing sum-to-one window rescaling option.
    """
    big = 2**24
    sacc_data = _windowed_sacc(np.float32, [[big, 1], [1, 2], [1, 3], [1, 4]])
    _, weights = extract_window_function(
        sacc_data, _window_indices(sacc_data), normalize=normalize
    )
    assert weights is not None
    raw = np.array([[big, 1], [1, 2], [1, 3], [1, 4]], dtype=np.float64)
    totals = np.array([big + 3, 10.0])
    if normalize:
        np.testing.assert_array_equal(weights, raw / totals)
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, rtol=1e-15)
    else:
        np.testing.assert_array_equal(weights, raw)
        np.testing.assert_array_equal(weights.sum(axis=0), totals)


@pytest.mark.parametrize("normalize", [True, False])
def test_zero_weight_check_uses_float64_sums(normalize: bool) -> None:
    """Reject a column that is zero in float64 although float32 summation hides it.

    :param normalize: Existing sum-to-one window rescaling option.
    """
    big = 2**24
    sacc_data = _windowed_sacc(np.float32, [[big, 1], [1, 2], [1, 3], [-(big + 2), 4]])
    with pytest.raises(ValueError, match="zero total weight.*\\[0\\]"):
        extract_window_function(
            sacc_data, _window_indices(sacc_data), normalize=normalize
        )


@pytest.mark.parametrize("normalize", [True, False])
def test_zero_weight_column_still_rejected(normalize: bool) -> None:
    """Retain the independent zero-weight validation for converted integer weights.

    :param normalize: Existing sum-to-one window rescaling option.
    """
    sacc_data = _windowed_sacc(np.int32, [[1, 0], [3, 0], [5, 0], [7, 0]])
    with pytest.raises(ValueError, match="zero total weight.*\\[1\\]"):
        extract_window_function(
            sacc_data, _window_indices(sacc_data), normalize=normalize
        )


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "bad", _rejected_arrays((4, 2))[:4], ids=["bool", "complex", "str", "object"]
)
def test_extract_window_function_rejects_unsupported_weights(
    bad: npt.NDArray[np.generic], normalize: bool
) -> None:
    """Reject unsupported weight families before they are summed or divided.

    SACC's window accessor returns a plain ndarray even for masked stored weights,
    so a mask cannot reach this boundary; the constructor tests cover masked input.

    :param bad: Deliberately unsupported weight representation.
    :param normalize: Existing sum-to-one window rescaling option.
    """
    sacc_data = _windowed_sacc(np.float64)
    window = sacc_data.data[0].get_tag("window")
    window.weight = bad
    with pytest.raises(TypeError, match="real integer or floating"):
        extract_window_function(
            sacc_data, _window_indices(sacc_data), normalize=normalize
        )


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("normalize", [True, False])
def test_windowed_extraction_dtype_roles(dtype: _RealDtype, normalize: bool) -> None:
    """Keep support integer, labels real and every physical array float64.

    :param dtype: Dtype of SACC weights, data and covariance.
    :param normalize: Existing sum-to-one window rescaling option.
    """
    sacc_data = _windowed_sacc(dtype)
    (measurement,) = extract_all_harmonic_data(sacc_data, normalize=normalize)
    (metadata,) = extract_all_harmonic_metadata(sacc_data, normalize=normalize)
    for meta in (measurement.metadata, metadata):
        assert isinstance(meta, TwoPointHarmonic)
        assert meta.ells.dtype == np.int64
        np.testing.assert_array_equal(meta.ells, SUPPORT)
        assert meta.window is not None and meta.window.dtype == np.float64
        assert meta.window_ells is not None and meta.window_ells.dtype == np.float64
        np.testing.assert_array_equal(meta.window_ells, LABELS)
    assert measurement.data.dtype == np.float64
    np.testing.assert_array_equal(measurement.data, [1.0, 2.0])
    assert measurement.indices.dtype == np.int64
    assert isinstance(measurement.metadata, TwoPointHarmonic)
    assert measurement.metadata.window is not None
    raw = np.array(WEIGHTS, dtype=np.float64)
    expected = raw / np.array(COLUMN_SUMS) if normalize else raw
    np.testing.assert_array_equal(measurement.metadata.window, expected)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_covariance_identifier_uses_source_representation(
    dtype: _RealDtype,
) -> None:
    """Hash the covariance buffer in its original SACC dtype, without migration.

    :param dtype: Dtype of the SACC covariance buffer.
    """
    sacc_data = _windowed_sacc(dtype)
    (measurement,) = extract_all_harmonic_data(sacc_data)
    original = np.diag(np.array([4, 9], dtype=dtype))
    assert measurement.covariance_name == hashlib.sha256(original).hexdigest()
    canonical = hashlib.sha256(original.astype(np.float64)).hexdigest()
    assert (measurement.covariance_name == canonical) == (dtype == np.float64)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_unwindowed_extraction_dtype_roles(dtype: _RealDtype) -> None:
    """Keep exact unwindowed multipoles integer while measurements become float64.

    :param dtype: Dtype of SACC measurement values and covariance.
    """
    sacc_data = sacc.Sacc()
    z = np.linspace(0.0, 1.0, 20)
    sacc_data.add_tracer("NZ", "src0", z, np.exp(-((z - 0.5) ** 2) / 0.02))
    for ell, value in zip((2, 5), np.array(DATA, dtype=dtype)):
        sacc_data.add_data_point("galaxy_shear_cl_ee", ("src0", "src0"), value, ell=ell)
    sacc_data.add_covariance(np.diag(np.array([4, 9], dtype=dtype)))
    (measurement,) = extract_all_harmonic_data(sacc_data)
    assert isinstance(measurement.metadata, TwoPointHarmonic)
    assert measurement.metadata.window is None
    assert measurement.metadata.ells.dtype == np.int64
    np.testing.assert_array_equal(measurement.metadata.ells, [2, 5])
    assert measurement.data.dtype == np.float64
    np.testing.assert_array_equal(measurement.data, [1.0, 2.0])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_unwindowed_fractional_coordinates_stay_real(dtype: _RealDtype) -> None:
    """Retain fractional unwindowed sampling coordinates as real float64 values.

    :param dtype: Dtype of SACC measurement values.
    """
    sacc_data = sacc.Sacc()
    z = np.linspace(0.0, 1.0, 20)
    sacc_data.add_tracer("NZ", "src0", z, np.exp(-((z - 0.5) ** 2) / 0.02))
    sacc_data.add_ell_cl(
        "galaxy_shear_cl_ee",
        "src0",
        "src0",
        np.array([2.5, 7.5]),
        np.array(DATA, dtype=dtype),
    )
    sacc_data.add_covariance(np.diag(np.array([4, 9], dtype=dtype)))
    (measurement,) = extract_all_harmonic_data(sacc_data)
    assert isinstance(measurement.metadata, TwoPointHarmonic)
    assert measurement.metadata.ells.dtype == np.float64
    np.testing.assert_array_equal(measurement.metadata.ells, [2.5, 7.5])
    assert measurement.data.dtype == np.float64


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_real_extraction_dtype_roles(dtype: _RealDtype) -> None:
    """Convert real coordinates and measurement values to float64 at extraction.

    :param dtype: Dtype of SACC measurement values and covariance.
    """
    sacc_data = sacc.Sacc()
    z = np.linspace(0.0, 1.0, 20)
    sacc_data.add_tracer("NZ", "src0", z, np.exp(-((z - 0.5) ** 2) / 0.02))
    # Integer theta values reach SACC as integers, so use the generic API.
    for theta, value in zip((10, 20), np.array(DATA, dtype=dtype)):
        sacc_data.add_data_point(
            "galaxy_shear_xi_plus", ("src0", "src0"), value, theta=theta
        )
    sacc_data.add_covariance(np.diag(np.array([4, 9], dtype=dtype)))
    (measurement,) = extract_all_real_data(sacc_data)
    (metadata,) = extract_all_real_metadata(sacc_data)
    assert isinstance(measurement.metadata, TwoPointReal)
    for thetas in (measurement.metadata.thetas, metadata.thetas):
        assert thetas.dtype == np.float64
        np.testing.assert_array_equal(thetas, [10.0, 20.0])
    assert measurement.data.dtype == np.float64
    np.testing.assert_array_equal(measurement.data, [1.0, 2.0])
    assert measurement.indices.dtype == np.int64
    original = np.diag(np.array([4, 9], dtype=dtype))
    assert measurement.covariance_name == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_two_point_harmonic_converts_window_arrays(
    harmonic_two_point_xy: TwoPointXY, dtype: _RealDtype
) -> None:
    """Store window weights and labels as float64, sharing conforming arrays.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param dtype: Caller dtype for the window weights and labels.
    """
    window = _real(WEIGHTS, dtype)
    labels = _real([3, 4], dtype)
    metadata = TwoPointHarmonic(
        XY=harmonic_two_point_xy, ells=SUPPORT, window=window, window_ells=labels
    )
    assert metadata.window is not None and metadata.window_ells is not None
    assert metadata.window.dtype == np.float64
    assert metadata.window_ells.dtype == np.float64
    assert metadata.ells.dtype == np.int64
    np.testing.assert_array_equal(metadata.window, np.array(WEIGHTS, dtype=np.float64))
    np.testing.assert_array_equal(metadata.window_ells, [3.0, 4.0])
    if dtype == np.float64:
        assert np.shares_memory(metadata.window, window)
        assert np.shares_memory(metadata.window_ells, labels)


@pytest.mark.parametrize(
    "bad", _rejected_arrays((4, 2)), ids=["bool", "complex", "str", "object", "masked"]
)
def test_two_point_harmonic_rejects_unsupported_window(
    harmonic_two_point_xy: TwoPointXY, bad: npt.NDArray[np.generic]
) -> None:
    """Reject unsupported window families before conversion can lose information.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param bad: Deliberately unsupported window representation.
    """
    with pytest.raises(TypeError, match="real integer or floating|masked"):
        TwoPointHarmonic(
            XY=harmonic_two_point_xy,
            ells=SUPPORT,
            window=bad,  # type: ignore[arg-type]  # Invalid runtime ingress.
            window_ells=np.array([3.5, 4.5]),
        )


@pytest.mark.parametrize(
    "bad", _rejected_arrays((2,)), ids=["bool", "complex", "str", "object", "masked"]
)
def test_two_point_harmonic_rejects_unsupported_labels(
    harmonic_two_point_xy: TwoPointXY, bad: npt.NDArray[np.generic]
) -> None:
    """Reject unsupported effective-label families.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param bad: Deliberately unsupported label representation.
    """
    with pytest.raises(TypeError, match="real integer or floating|masked"):
        TwoPointHarmonic(
            XY=harmonic_two_point_xy,
            ells=SUPPORT,
            window=np.array(WEIGHTS, dtype=np.float64),
            window_ells=bad,  # type: ignore[arg-type]  # Invalid runtime ingress.
        )


def test_two_point_harmonic_shape_validation_unchanged(
    harmonic_two_point_xy: TwoPointXY,
) -> None:
    """Keep independent window shape validation reachable after conversion.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    """
    with pytest.raises(ValueError, match="window should be a 2D array."):
        TwoPointHarmonic(
            XY=harmonic_two_point_xy, ells=SUPPORT, window=np.ones(8, dtype=np.int32)
        )
    with pytest.raises(ValueError, match="same number of rows as ells"):
        TwoPointHarmonic(
            XY=harmonic_two_point_xy,
            ells=SUPPORT[:3],
            window=np.ones((4, 2), dtype=np.float32),
            window_ells=np.array([1.0, 2.0]),
        )


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_two_point_real_converts_thetas(
    optimized_real_two_point_xy: TwoPointXY, dtype: _RealDtype
) -> None:
    """Store real angular coordinates as float64, sharing conforming arrays.

    :param optimized_real_two_point_xy: Compatible real-space tracer pair.
    :param dtype: Caller dtype for the coordinates.
    """
    thetas = _real([1, 2, 3], dtype)
    metadata = TwoPointReal(XY=optimized_real_two_point_xy, thetas=thetas)
    assert metadata.thetas.dtype == np.float64
    np.testing.assert_array_equal(metadata.thetas, [1.0, 2.0, 3.0])
    if dtype == np.float64:
        assert np.shares_memory(metadata.thetas, thetas)


@pytest.mark.parametrize(
    "bad", _rejected_arrays((3,)), ids=["bool", "complex", "str", "object", "masked"]
)
def test_two_point_real_rejects_unsupported_thetas(
    optimized_real_two_point_xy: TwoPointXY, bad: npt.NDArray[np.generic]
) -> None:
    """Reject unsupported angular-coordinate families.

    :param optimized_real_two_point_xy: Compatible real-space tracer pair.
    :param bad: Deliberately unsupported coordinate representation.
    """
    with pytest.raises(TypeError, match="real integer or floating|masked"):
        TwoPointReal(
            XY=optimized_real_two_point_xy,
            thetas=bad,  # type: ignore[arg-type]  # Invalid runtime ingress.
        )


def test_two_point_real_shape_validation_unchanged(
    optimized_real_two_point_xy: TwoPointXY,
) -> None:
    """Keep the independent one-dimensional coordinate check reachable.

    :param optimized_real_two_point_xy: Compatible real-space tracer pair.
    """
    with pytest.raises(ValueError, match="Thetas should be a 1D array."):
        TwoPointReal(
            XY=optimized_real_two_point_xy, thetas=np.ones((2, 2), dtype=np.int32)
        )


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_measurement_converts_data(
    harmonic_two_point_xy: TwoPointXY, dtype: _RealDtype
) -> None:
    """Store measured values as float64, sharing conforming arrays.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param dtype: Caller dtype for the measured values.
    """
    data = _real([3, 4], dtype)
    measurement = TwoPointMeasurement(
        data=data,
        indices=np.array([0, 1], dtype=np.int64),
        covariance_name="cov",
        metadata=TwoPointHarmonic(
            XY=harmonic_two_point_xy, ells=np.array([2, 5], dtype=np.int64)
        ),
    )
    assert measurement.data.dtype == np.float64
    np.testing.assert_array_equal(measurement.data, [3.0, 4.0])
    assert measurement.indices.dtype == np.int64
    if dtype == np.float64:
        assert np.shares_memory(measurement.data, data)


@pytest.mark.parametrize(
    "bad", _rejected_arrays((2,)), ids=["bool", "complex", "str", "object", "masked"]
)
def test_measurement_rejects_unsupported_data(
    harmonic_two_point_xy: TwoPointXY, bad: npt.NDArray[np.generic]
) -> None:
    """Reject unsupported measured-value families.

    :param harmonic_two_point_xy: Compatible harmonic tracer pair.
    :param bad: Deliberately unsupported data representation.
    """
    with pytest.raises(TypeError, match="real integer or floating|masked"):
        TwoPointMeasurement(
            data=bad,  # type: ignore[arg-type]  # Invalid runtime ingress.
            indices=np.array([0, 1], dtype=np.int64),
            covariance_name="cov",
            metadata=TwoPointHarmonic(
                XY=harmonic_two_point_xy, ells=np.array([2, 5], dtype=np.int64)
            ),
        )


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_filtering_keeps_float64_values_and_boolean_masks(
    dtype: _RealDtype,
) -> None:
    """Filter converted windows with Boolean masks, retaining labels and integers.

    :param dtype: Dtype of SACC weights, data and covariance.
    """
    sacc_data = _windowed_sacc(dtype)
    tpms = extract_all_harmonic_data(sacc_data)
    label_cut = TwoPointBinFilter.from_args(
        name1="src0",
        name2="src0",
        measurement1=Galaxies.SHEAR_E,
        measurement2=Galaxies.SHEAR_E,
        lower=0.0,
        upper=4.0,
        method=TwoPointFilterMethod.LABEL,
    )
    collection = TwoPointBinFilterCollection(
        require_filter_for_all=True, allow_empty=False, filters=[label_cut]
    )
    match_elements, match_obs = collection.apply_filter_single(tpms[0])
    assert match_elements.dtype == np.bool_ and match_obs.dtype == np.bool_
    np.testing.assert_array_equal(match_elements, [True] * 4)
    np.testing.assert_array_equal(match_obs, [True, False])

    (kept,) = collection(tpms)
    assert isinstance(kept.metadata, TwoPointHarmonic)
    assert kept.data.dtype == np.float64
    np.testing.assert_array_equal(kept.data, [1.0])
    assert kept.metadata.window is not None
    assert kept.metadata.window.dtype == np.float64
    np.testing.assert_array_equal(
        kept.metadata.window, np.array([[1.0], [3.0], [5.0], [7.0]]) / 16.0
    )
    assert kept.metadata.window_ells is not None
    assert kept.metadata.window_ells.dtype == np.float64
    np.testing.assert_array_equal(kept.metadata.window_ells, [3.5])
    assert kept.metadata.ells.dtype == np.int64
    assert kept.indices.dtype == np.int64


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize(
    ("method", "upper", "expected_kept"),
    [
        (TwoPointFilterMethod.SUPPORT, 5.0, [True, True]),
        (TwoPointFilterMethod.SUPPORT, 4.0, [False, False]),
        (TwoPointFilterMethod.SUPPORT_95, 5.0, [True, True]),
    ],
)
def test_support_cuts_use_converted_weights(
    dtype: _RealDtype,
    method: TwoPointFilterMethod,
    upper: float,
    expected_kept: list[bool],
) -> None:
    """Keep the existing cumulative-support scale-cut semantics for converted weights.

    :param dtype: Dtype of SACC weights, data and covariance.
    :param method: Existing support-based filter method.
    :param upper: Upper multipole bound of the cut.
    :param expected_kept: Independent expectation of retained bandpowers.
    """
    tpms = extract_all_harmonic_data(_windowed_sacc(dtype))
    support_cut = TwoPointBinFilter.from_args(
        name1="src0",
        name2="src0",
        measurement1=Galaxies.SHEAR_E,
        measurement2=Galaxies.SHEAR_E,
        lower=0.0,
        upper=upper,
        method=method,
    )
    collection = TwoPointBinFilterCollection(
        require_filter_for_all=True, allow_empty=True, filters=[support_cut]
    )
    _, match_obs = collection.apply_filter_single(tpms[0])
    assert match_obs.dtype == np.bool_
    np.testing.assert_array_equal(match_obs, expected_kept)
