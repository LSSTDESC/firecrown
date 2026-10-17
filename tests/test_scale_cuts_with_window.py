"""Test that the scale cuts with bandpower windows work correctly."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyccl as ccl
import pytest
import sacc

from firecrown.data_functions import (
    TwoPointBinFilter,
    TwoPointBinFilterCollection,
)
from firecrown.likelihood import ConstGaussian, TwoPointFactory
from firecrown.likelihood.factories import (
    DataSourceSacc,
    TwoPointExperiment,
)
from firecrown.metadata_types import Galaxies, TwoPointFilterMethod
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import ParamsMap, get_default_params
from firecrown.utils import base_model_from_yaml, upper_triangle_indices


@dataclass(frozen=True, kw_only=True)
class ScaleCutWindowFunction:
    """Window weights, support multipoles, and geometric bin centers.

    Frozen bindings leave the arrays mutable. Names and dtype annotations do not
    enforce matrix dimensions, shapes, same-dtype array swaps, or physical units.

    :param weights: Column-normalized weights, one row per support multipole and
        one column per bin.
    :param support_ells: Integer multipoles on which the window is sampled.
    :param bin_centers: Geometric means of adjacent bin edges, used as SACC
        observation labels; not weighted means of the window support.
    """

    weights: npt.NDArray[np.float64]
    support_ells: npt.NDArray[np.int64]
    bin_centers: npt.NDArray[np.float64]


@pytest.fixture(name="window_function_data", scope="module")
def fixture_window_function() -> ScaleCutWindowFunction:
    """Generate a normalized window on 800 multipoles for five bins.

    :returns: Window weights, support multipoles, and geometric bin centers.
    """
    n_ell_bin = 5
    ell_bin_edges = np.geomspace(10, 500, n_ell_bin + 1)
    ell_bin_centers: npt.NDArray[np.float64] = np.sqrt(
        ell_bin_edges[:-1] * ell_bin_edges[1:]
    )
    ell_bin_widths = np.diff(ell_bin_edges)

    ell_window: npt.NDArray[np.int64] = np.arange(800)
    window_function: npt.NDArray[np.float64] = np.zeros(
        (ell_window.shape[0], n_ell_bin)
    )
    for i, (mu_ell, sigma_ell) in enumerate(zip(ell_bin_centers, ell_bin_widths)):
        window_function[:, i] = np.exp(
            -0.5 * (ell_window - mu_ell) ** 2 / (sigma_ell) ** 2
        )

    window_function = window_function / window_function.sum(axis=0, keepdims=True)

    return ScaleCutWindowFunction(
        weights=window_function,
        support_ells=ell_window,
        bin_centers=ell_bin_centers,
    )


@dataclass(frozen=True, kw_only=True)
class ScaleCutBandpowerPrediction:
    """Independent CCL bandpower predictions and their scale-cut window context.

    :param bandpowers: CCL predictions projected through the normalized window,
        one value per bin, also used as the fixture's SACC measurements.
    :param window: Shared normalized weights, support multipoles, and geometric
        bin centers; frozen bindings do not freeze its arrays or check shapes.
    """

    bandpowers: npt.NDArray[np.float64]
    window: ScaleCutWindowFunction


@dataclass(frozen=True, kw_only=True)
class ScaleCutSaccFixture:
    """Cosmic-shear SACC data with independent predictions and their cosmology.

    Frozen bindings leave SACC, the spectrum mapping, and nested arrays mutable.
    Names and dtype annotations do not check same-dtype swaps, shapes, or units.

    :param sacc_data: Windowed SACC measurements for production likelihood loading.
    :param spectra: Independent CCL predictions keyed by ordered tracer-pair names.
    :param cosmology: Cosmology used to generate the CCL tracers and predictions.
    """

    sacc_data: sacc.Sacc
    spectra: dict[str, ScaleCutBandpowerPrediction]
    cosmology: ccl.Cosmology


@pytest.fixture(name="minimal_3x2pt_sacc", scope="module")
def fixture_minimal_3x2pt_sacc(
    window_function_data: ScaleCutWindowFunction,
) -> ScaleCutSaccFixture:
    """Generate SACC data with window functions for cosmic shear only.

    :param window_function_data: Window weights, support, and observation centers.
    :returns: SACC data, independent CCL bandpowers with window data, and cosmology.
    """
    sacc_data = sacc.Sacc()

    z = (np.linspace(0, 4.0, 256) + 0.05).astype(np.float64)

    src_bins_centers = np.linspace(0.25, 0.75, 2)

    ccl_tracers: dict[str, ccl.WeakLensingTracer] = {}
    ccl_cell: dict[str, ScaleCutBandpowerPrediction] = {}
    cosmo = ccl.CosmologyVanillaLCDM()

    for i, mn in enumerate(src_bins_centers):
        dndz: npt.NDArray[np.float64] = np.exp(-0.5 * (z - mn) ** 2 / 0.05**2)
        sacc_data.add_tracer("NZ", f"src{i}", z, dndz)
        ccl_tracers[f"src{i}"] = ccl.WeakLensingTracer(
            cosmo=cosmo,
            dndz=(z, dndz),
        )

    for i, j in upper_triangle_indices(len(src_bins_centers)):
        window = sacc.BandpowerWindow(
            window_function_data.support_ells, window_function_data.weights
        )
        Cell: npt.NDArray[np.float64] = ccl.angular_cl(
            cosmo=cosmo,
            tracer1=ccl_tracers[f"src{i}"],
            tracer2=ccl_tracers[f"src{j}"],
            ell=window_function_data.support_ells,
        )
        Cell_binned: npt.NDArray[np.float64] = window_function_data.weights.T @ Cell
        sacc_data.add_ell_cl(
            data_type="galaxy_shear_cl_ee",
            tracer1=f"src{i}",
            tracer2=f"src{j}",
            ell=window_function_data.bin_centers,
            x=Cell_binned,
            window=window,
        )
        ccl_cell[f"src{i}-src{j}"] = ScaleCutBandpowerPrediction(
            bandpowers=Cell_binned,
            window=window_function_data,
        )

    sacc_data.add_covariance(np.identity(len(sacc_data)) * 0.01)

    return ScaleCutSaccFixture(sacc_data=sacc_data, spectra=ccl_cell, cosmology=cosmo)


@pytest.fixture(
    name="cut",
    params=[
        ("src0", "src0", 100),
        ("src0", "src1", 80),
        ("src1", "src1", 300),
        ("src0", "src0", 60),
        ("src0", "src1", 160),
        ("src1", "src1", 200),
    ],
)
def fixture_cut(request: pytest.FixtureRequest) -> tuple[str, str, int]:
    return request.param


two_point_yaml = """
correlation_space: harmonic
weak_lensing_factories:
  - type_source: default
    per_bin_systematics: []
    global_systematics: []
"""


def test_scale_cuts_with_bandpower_window_label(
    minimal_3x2pt_sacc: ScaleCutSaccFixture,
    cut: tuple[str, str, int],
    tmp_path: Path,
) -> None:
    """Test scale cuts selected by bandpower window labels.

    :param minimal_3x2pt_sacc: SACC data, independent CCL predictions, and cosmology.
    :param cut: Ordered tracer names and upper multipole threshold.
    :param tmp_path: Directory for the SACC file loaded by the likelihood factory.
    """
    sacc_data = minimal_3x2pt_sacc.sacc_data
    ccl_cell = minimal_3x2pt_sacc.spectra
    cosmo = minimal_3x2pt_sacc.cosmology

    tp_factory = base_model_from_yaml(TwoPointFactory, two_point_yaml)

    tmp_file_sacc = tmp_path / "sacc_data.fits"
    sacc_data.save_fits(tmp_file_sacc.as_posix(), overwrite=True)

    two_point_experiment = TwoPointExperiment(
        two_point_factory=tp_factory,
        data_source=DataSourceSacc(
            sacc_data_file=tmp_file_sacc.as_posix(),
            filters=TwoPointBinFilterCollection(
                require_filter_for_all=False,
                allow_empty=True,
                filters=[
                    TwoPointBinFilter.from_args(
                        name1=cut[0],
                        name2=cut[1],
                        measurement1=Galaxies.SHEAR_E,
                        measurement2=Galaxies.SHEAR_E,
                        lower=0.0,
                        upper=cut[2],
                        method=TwoPointFilterMethod.LABEL,
                    )
                ],
            ),
        ),
    )
    likelihood = two_point_experiment.make_likelihood()

    tools = ModelingTools(
        ccl_factory=two_point_experiment.ccl_factory,
    )
    params_dict = get_default_params(tools, likelihood)
    params_dict.update(
        {k: v for k, v in cosmo.to_dict().items() if isinstance(v, float)}
    )
    params = ParamsMap(params_dict)
    likelihood.update(params)
    tools.update(params)
    tools.prepare()

    log_like = likelihood.compute_loglike(tools)
    assert np.isclose(log_like, 0.0)
    assert isinstance(likelihood, ConstGaussian)

    assert np.allclose(likelihood.get_data_vector(), likelihood.get_theory_vector())

    ccl_cell_cut_data_vector = np.concatenate(
        [
            (
                spectrum.bandpowers
                if name != f"{cut[0]}-{cut[1]}"
                else spectrum.bandpowers[spectrum.window.bin_centers < cut[2]]
            )
            for name, spectrum in ccl_cell.items()
        ]
    )
    # Check that the shapes match (i.e. that the correct data points are selected)
    assert ccl_cell_cut_data_vector.shape == likelihood.get_theory_vector().shape

    # Check that the computation of the theory vector matches the CCL computation
    assert np.allclose(likelihood.get_theory_vector(), ccl_cell_cut_data_vector)


@pytest.mark.parametrize(
    "method",
    [
        (TwoPointFilterMethod.SUPPORT, 0.999),
        (TwoPointFilterMethod.SUPPORT_95, 0.95),
    ],
)
def test_scale_cuts_with_bandpower_window_support(
    minimal_3x2pt_sacc: ScaleCutSaccFixture,
    cut: tuple[str, str, int],
    method: tuple[TwoPointFilterMethod, float],
    tmp_path: Path,
) -> None:
    """Test scale cuts selected by cumulative bandpower window support.

    :param minimal_3x2pt_sacc: SACC data, independent CCL predictions, and cosmology.
    :param cut: Ordered tracer names and upper multipole threshold.
    :param method: Filter method and required fraction of window support.
    :param tmp_path: Directory for the SACC file loaded by the likelihood factory.
    """
    sacc_data = minimal_3x2pt_sacc.sacc_data
    ccl_cell = minimal_3x2pt_sacc.spectra
    cosmo = minimal_3x2pt_sacc.cosmology

    tp_factory = base_model_from_yaml(TwoPointFactory, two_point_yaml)

    tmp_file_sacc = tmp_path / "sacc_data.fits"
    sacc_data.save_fits((tmp_path / "sacc_data.fits").as_posix(), overwrite=True)

    two_point_experiment = TwoPointExperiment(
        two_point_factory=tp_factory,
        data_source=DataSourceSacc(
            sacc_data_file=tmp_file_sacc.as_posix(),
            filters=TwoPointBinFilterCollection(
                require_filter_for_all=False,
                allow_empty=True,
                filters=[
                    TwoPointBinFilter.from_args(
                        name1=cut[0],
                        name2=cut[1],
                        measurement1=Galaxies.SHEAR_E,
                        measurement2=Galaxies.SHEAR_E,
                        lower=0.0,
                        upper=cut[2],
                        method=method[0],
                    )
                ],
            ),
        ),
    )
    likelihood = two_point_experiment.make_likelihood()

    tools = ModelingTools(
        ccl_factory=two_point_experiment.ccl_factory,
    )
    params = ParamsMap(
        get_default_params(tools, likelihood)
        | {k: v for k, v in cosmo.to_dict().items() if isinstance(v, float)}
    )

    likelihood.update(params)
    tools.update(params)
    tools.prepare()

    log_like = likelihood.compute_loglike(tools)
    assert np.isclose(log_like, 0.0)
    assert isinstance(likelihood, ConstGaussian)

    assert np.allclose(likelihood.get_data_vector(), likelihood.get_theory_vector())

    ccl_cell_cut_data_vector = np.concatenate(
        [
            (
                spectrum.bandpowers
                if name != f"{cut[0]}-{cut[1]}"
                else spectrum.bandpowers[
                    spectrum.window.weights[spectrum.window.support_ells < cut[2]].sum(
                        axis=0
                    )
                    >= method[1]
                ]
            )
            for name, spectrum in ccl_cell.items()
        ]
    )
    # Check that the shapes match (i.e. that the correct data points are selected)
    assert ccl_cell_cut_data_vector.shape == likelihood.get_theory_vector().shape

    # Check that the computation of the theory vector matches the CCL computation
    assert np.allclose(likelihood.get_theory_vector(), ccl_cell_cut_data_vector)
