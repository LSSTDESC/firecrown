"""SACC-to-likelihood workflow with windows and data at several real dtypes.

Each case runs real pyccl through extraction/read, update, prepare and compute.
Expectations come from independent float64 pyccl calls, never from a Firecrown
converter.
"""

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pyccl
import pytest
import sacc

import firecrown.likelihood._two_point as tp
from firecrown.data_functions import TwoPointBinFilter, TwoPointBinFilterCollection
from firecrown.likelihood import ConstGaussian, TwoPointFactory
from firecrown.likelihood.factories import DataSourceSacc, TwoPointExperiment
from firecrown.likelihood.weak_lensing import WeakLensing
from firecrown.metadata_types import Galaxies, TwoPointFilterMethod
from firecrown.modeling_tools import ModelingTools
from firecrown.models.two_point import ApplyInterpolationWhen
from firecrown.updatable import ParamsMap, get_default_params
from firecrown.utils import base_model_from_yaml

_RealDtype = type[np.float64] | type[np.float32] | type[np.int64]

DTYPES = [np.float64, np.float32, np.int64]
SUPPORT = np.arange(2, 202, dtype=np.int64)
LABELS = np.array([20.5, 80.5, 150.5])
Z = np.linspace(0.0, 2.0, 50)
DNDZ = np.exp(-0.5 * (Z - 0.5) ** 2 / 0.1**2)


def _integer_weights() -> npt.NDArray[np.int64]:
    """Return a window whose integer values are scientifically valid weights.

    :returns: Support-by-bandpower weights with different column totals.
    """
    weights = np.zeros((SUPPORT.size, 3), dtype=np.int64)
    for column, (low, high, scale) in enumerate(
        [(2, 60, 3), (60, 120, 5), (120, 202, 7)]
    ):
        in_band = (SUPPORT >= low) & (SUPPORT < high)
        weights[in_band, column] = scale * (1 + (SUPPORT[in_band] % 4))
    return weights


def _float32_exact_weights() -> npt.NDArray[np.float32]:
    """Return weights chosen so float32 storage is exact, with a float32 sum error.

    :returns: Weights with distinct, large column totals.
    """
    weights = _integer_weights().astype(np.float32) * np.float32(0.25)
    weights[0, 0] = np.float32(2**23)
    return weights


def _weights(dtype: _RealDtype) -> npt.NDArray[np.generic]:
    """Return the window weights stored at a given dtype.

    :param dtype: Storage dtype.
    :returns: Weights exactly representable at that dtype.
    """
    if dtype == np.float32:
        return _float32_exact_weights()
    if dtype == np.int64:
        return _integer_weights()
    return _float32_exact_weights().astype(np.float64)


def _sacc_from_arrays(
    weights: npt.NDArray[np.generic],
    values: npt.NDArray[np.generic],
    variances: npt.NDArray[np.generic],
) -> sacc.Sacc:
    """Build a windowed shear SACC from arrays stored exactly as given.

    :param weights: Window weights, one row per support multipole.
    :param values: Measured bandpowers.
    :param variances: Diagonal covariance entries.
    :returns: SACC data keeping every input dtype.
    """
    sacc_data = sacc.Sacc()
    sacc_data.add_tracer("NZ", "src0", Z, DNDZ)
    sacc_data.add_ell_cl(
        "galaxy_shear_cl_ee",
        "src0",
        "src0",
        LABELS,
        values,
        window=sacc.BandpowerWindow(SUPPORT, weights),
    )
    sacc_data.add_covariance(np.diag(variances))
    return sacc_data


def _windowed_sacc(dtype: _RealDtype) -> sacc.Sacc:
    """Build windowed shear SACC data whose arrays are stored at one dtype.

    :param dtype: Dtype of window weights, measured values and covariance.
    :returns: SACC data with exactly representable weights and values.
    """
    return _sacc_from_arrays(
        _weights(dtype),
        np.array([3, 5, 7], dtype=dtype),
        np.array([4, 9, 16], dtype=dtype),
    )


def _independent_bandpowers(
    normalize: bool, cosmo: pyccl.Cosmology
) -> npt.NDArray[np.float64]:
    """Compute window-contracted spectra with float64 arithmetic only.

    :param normalize: Whether to divide each column by its float64 total.
    :param cosmo: Cosmology used by the Firecrown likelihood.
    :returns: The expected theory vector.
    """
    tracer = pyccl.WeakLensingTracer(cosmo, dndz=(Z, DNDZ))
    cells = pyccl.angular_cl(cosmo, tracer, tracer, ell=SUPPORT.astype(np.float64))
    raw = _float32_exact_weights().astype(np.float64)
    return (raw / raw.sum(axis=0) if normalize else raw).T @ cells


def _evaluate(
    statistic: tp.TwoPoint, sacc_data: sacc.Sacc
) -> tuple[ConstGaussian, ModelingTools, npt.NDArray[np.float64]]:
    """Read, update, prepare and compute the theory vector for a statistic.

    :param statistic: The statistic to evaluate.
    :param sacc_data: Data and covariance to read.
    :returns: The likelihood, its tools and the computed theory vector.
    """
    likelihood = ConstGaussian([statistic])
    likelihood.read(sacc_data)
    tools = ModelingTools()
    params = ParamsMap(get_default_params(tools, likelihood))
    likelihood.update(params)
    tools.update(params)
    tools.prepare()
    likelihood.compute_loglike(tools)
    return likelihood, tools, likelihood.get_theory_vector()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("normalize", [True, False])
def test_windowed_read_workflow(dtype: _RealDtype, normalize: bool) -> None:
    """Match independent float64 bandpowers for every ingress dtype and both modes.

    Float32 weights whose float32 column sum is inexact would shift the normalized
    result if summation preceded conversion; the tight tolerance would catch that.

    :param dtype: Dtype of window weights and measurements.
    :param normalize: Existing sum-to-one window rescaling option.
    """
    sacc_data = _windowed_sacc(dtype)
    statistic = tp.TwoPoint(
        "galaxy_shear_cl_ee",
        WeakLensing(sacc_tracer="src0"),
        WeakLensing(sacc_tracer="src0"),
        apply_interp=ApplyInterpolationWhen.NONE,
        normalize_window=normalize,
    )
    likelihood, tools, theory = _evaluate(statistic, sacc_data)

    window = statistic.theory.window
    assert window is not None and window.dtype == np.float64
    assert statistic.theory.ells is not None and statistic.theory.ells.dtype == np.int64
    assert statistic.sacc_indices is not None
    assert statistic.sacc_indices.dtype == np.int64
    assert statistic.get_data_vector().dtype == np.float64
    assert theory.dtype == np.float64
    assert likelihood.get_data_vector().dtype == np.float64
    np.testing.assert_array_equal(likelihood.get_data_vector(), [3.0, 5.0, 7.0])

    if dtype == np.int64:
        # Integer weights are a different (scientifically valid) window.
        raw = _integer_weights().astype(np.float64)
        np.testing.assert_array_equal(
            window, raw / raw.sum(axis=0) if normalize else raw
        )
        return
    expected = _independent_bandpowers(normalize, tools.get_ccl_cosmology())
    np.testing.assert_allclose(theory, expected, rtol=1e-12, atol=0.0)
    if normalize:
        np.testing.assert_allclose(window.sum(axis=0), 1.0, rtol=1e-15)
    else:
        raw = _float32_exact_weights().astype(np.float64)
        np.testing.assert_array_equal(window, raw)
    mean_ells = statistic.theory.mean_ells
    assert mean_ells is not None and mean_ells.dtype == np.float64
    np.testing.assert_allclose(
        mean_ells, window.T @ SUPPORT.astype(np.float64), rtol=1e-15
    )


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
@pytest.mark.parametrize(
    "interp", [ApplyInterpolationWhen.NONE, ApplyInterpolationWhen.DEFAULT]
)
def test_windowed_lower_precision_matches_float64_input(
    dtype: _RealDtype, interp: ApplyInterpolationWhen
) -> None:
    """Match the result for the same values stored as float64.

    Covers both the exact and the interpolated window computation.

    :param dtype: Lower-precision window and data dtype.
    :param interp: Whether the existing window interpolation is applied.
    """
    native = _windowed_sacc(dtype)
    reference = _sacc_from_arrays(
        _weights(dtype).astype(np.float64),
        np.array([3.0, 5.0, 7.0]),
        np.array([4.0, 9.0, 16.0]),
    )
    results = []
    for sacc_data in (native, reference):
        statistic = tp.TwoPoint(
            "galaxy_shear_cl_ee",
            WeakLensing(sacc_tracer="src0"),
            WeakLensing(sacc_tracer="src0"),
            apply_interp=interp,
        )
        results.append(_evaluate(statistic, sacc_data)[2])
    np.testing.assert_allclose(results[0], results[1], rtol=1e-12, atol=0.0)


YAML = """
correlation_space: harmonic
weak_lensing_factories:
  - type_source: default
    per_bin_systematics: []
    global_systematics: []
"""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("normalize", [True, False])
def test_windowed_experiment_workflow(
    dtype: _RealDtype, normalize: bool, tmp_path: Path
) -> None:
    """Run extraction, filtering, ready construction and evaluation from a file.

    :param dtype: Dtype of window weights and measurements.
    :param normalize: Existing sum-to-one window rescaling option.
    :param tmp_path: Directory for the saved SACC file.
    """
    path = tmp_path / "sacc.fits"
    _windowed_sacc(dtype).save_fits(str(path), overwrite=True)
    label_cut = TwoPointBinFilter.from_args(
        name1="src0",
        name2="src0",
        measurement1=Galaxies.SHEAR_E,
        measurement2=Galaxies.SHEAR_E,
        lower=0.0,
        upper=100.0,
        method=TwoPointFilterMethod.LABEL,
    )
    experiment = TwoPointExperiment(
        two_point_factory=base_model_from_yaml(TwoPointFactory, YAML),
        data_source=DataSourceSacc(
            sacc_data_file=str(path),
            normalize_window=normalize,
            filters=TwoPointBinFilterCollection(
                require_filter_for_all=True, allow_empty=False, filters=[label_cut]
            ),
        ),
    )
    likelihood = experiment.make_likelihood()
    tools = ModelingTools(ccl_factory=experiment.ccl_factory)
    params = ParamsMap(get_default_params(tools, likelihood))
    likelihood.update(params)
    tools.update(params)
    tools.prepare()
    likelihood.compute_loglike(tools)
    assert isinstance(likelihood, ConstGaussian)

    # Labels 20.5 and 80.5 satisfy the label cut; the 150.5 bandpower is removed.
    np.testing.assert_array_equal(likelihood.get_data_vector(), [3.0, 5.0])
    assert likelihood.get_data_vector().dtype == np.float64
    assert likelihood.get_theory_vector().dtype == np.float64
    assert likelihood.get_cov().dtype == np.float64
    np.testing.assert_array_equal(likelihood.get_cov(), np.diag([4.0, 9.0]))
    (statistic,) = likelihood.statistics
    two_point = cast(tp.TwoPoint, statistic.statistic)
    window = two_point.window
    assert window is not None
    assert window.dtype == np.float64 and window.shape == (SUPPORT.size, 2)
    assert two_point.sacc_indices is not None
    assert two_point.sacc_indices.dtype == np.int64
    if dtype != np.int64:
        expected = _independent_bandpowers(normalize, tools.get_ccl_cosmology())[:2]
        # The factory default interpolates window multipoles (measured error about
        # 5e-7 for this fixture), so this is looser than the exact-grid cases above.
        np.testing.assert_allclose(
            likelihood.get_theory_vector(), expected, rtol=2e-6, atol=0.0
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_unwindowed_workflow_with_real_data(dtype: _RealDtype) -> None:
    """Evaluate unwindowed data stored at several dtypes against float64 pyccl.

    :param dtype: Dtype of measured values and covariance.
    """
    ells = np.array([10, 40, 90], dtype=np.int64)
    sacc_data = sacc.Sacc()
    sacc_data.add_tracer("NZ", "src0", Z, DNDZ)
    for ell, value in zip(ells, np.array([3, 5, 7], dtype=dtype)):
        sacc_data.add_data_point("galaxy_shear_cl_ee", ("src0", "src0"), value, ell=ell)
    sacc_data.add_covariance(np.diag(np.array([4, 9, 16], dtype=dtype)))
    statistic = tp.TwoPoint(
        "galaxy_shear_cl_ee",
        WeakLensing(sacc_tracer="src0"),
        WeakLensing(sacc_tracer="src0"),
        apply_interp=ApplyInterpolationWhen.NONE,
    )
    likelihood, tools, theory = _evaluate(statistic, sacc_data)
    assert statistic.window is None
    assert statistic.theory.ells is not None and statistic.theory.ells.dtype == np.int64
    assert theory.dtype == np.float64
    np.testing.assert_array_equal(likelihood.get_data_vector(), [3.0, 5.0, 7.0])
    cosmo = tools.get_ccl_cosmology()
    tracer = pyccl.WeakLensingTracer(cosmo, dndz=(Z, DNDZ))
    expected = pyccl.angular_cl(cosmo, tracer, tracer, ell=ells.astype(np.float64))
    np.testing.assert_allclose(theory, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_real_space_workflow_converts_thetas(dtype: _RealDtype) -> None:
    """Evaluate real-space data with theta coordinates stored at several dtypes.

    :param dtype: Dtype of angular coordinates, measured values and covariance.
    """
    thetas = np.array([10, 20, 40], dtype=dtype)
    sacc_data = sacc.Sacc()
    sacc_data.add_tracer("NZ", "src0", Z, DNDZ)
    for theta, value in zip(thetas, np.array([3, 5, 7], dtype=dtype)):
        sacc_data.add_data_point(
            "galaxy_shear_xi_plus", ("src0", "src0"), value, theta=theta
        )
    sacc_data.add_covariance(np.diag(np.array([4, 9, 16], dtype=dtype)))
    statistic = tp.TwoPoint(
        "galaxy_shear_xi_plus",
        WeakLensing(sacc_tracer="src0"),
        WeakLensing(sacc_tracer="src0"),
    )
    likelihood, tools, theory = _evaluate(statistic, sacc_data)
    assert statistic.theory.thetas is not None
    assert statistic.theory.thetas.dtype == np.float64
    np.testing.assert_array_equal(statistic.theory.thetas, [10.0, 20.0, 40.0])
    assert theory.dtype == np.float64
    np.testing.assert_array_equal(likelihood.get_data_vector(), [3.0, 5.0, 7.0])
    cosmo = tools.get_ccl_cosmology()
    tracer = pyccl.WeakLensingTracer(cosmo, dndz=(Z, DNDZ))
    # Same multipole grid as the statistic's default real-space transform, so the
    # comparison needs no looser tolerance than the other float64 workflow tests.
    ells = statistic.theory.ells_for_xi.astype(np.float64)
    cells = pyccl.angular_cl(cosmo, tracer, tracer, ell=ells)
    expected = pyccl.correlation(
        cosmo,
        ell=ells,
        C_ell=cells,
        theta=np.array([10.0, 20.0, 40.0]) / 60.0,
        type="GG+",
    )
    np.testing.assert_allclose(theory, expected, rtol=1e-12, atol=0.0)
