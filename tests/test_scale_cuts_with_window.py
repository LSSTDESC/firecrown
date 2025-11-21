"""Test that the scale cuts with bandpower windows work correctly."""

from pathlib import Path
import pytest
import numpy as np

import pyccl as ccl

import sacc

from firecrown.likelihood import ConstGaussian
from firecrown.likelihood import TwoPointFactory
from firecrown.likelihood.factories import (
    DataSourceSacc,
    TwoPointExperiment,
)
from firecrown.data_functions import (
    TwoPointBinFilterCollection,
    TwoPointBinFilter,
)
from firecrown.metadata_types import Galaxies, TwoPointFilterMethod
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from firecrown.updatable import get_default_params
from firecrown.utils import base_model_from_yaml, upper_triangle_indices


@pytest.fixture(name="window_function_data", scope="module")
def fixture_window_function():
    """Generate window function data."""
    n_ell_bin = 5
    ell_bin_edges = np.geomspace(10, 500, n_ell_bin + 1)
    ell_bin_centers = np.sqrt(ell_bin_edges[:-1] * ell_bin_edges[1:])
    ell_bin_widths = np.diff(ell_bin_edges)

    ell_window = np.arange(800)
    window_function = np.zeros((ell_window.shape[0], n_ell_bin))
    for i, (mu_ell, sigma_ell) in enumerate(zip(ell_bin_centers, ell_bin_widths)):
        window_function[:, i] = np.exp(
            -0.5 * (ell_window - mu_ell) ** 2 / (sigma_ell) ** 2
        )

    window_function = window_function / window_function.sum(axis=0, keepdims=True)

    return window_function, ell_window, ell_bin_centers


@pytest.fixture(name="minimal_3x2pt_sacc", scope="module")
def fixture_minimal_3x2pt_sacc(window_function_data):
    """Generate SACC data with window functions.
    At the moment, this only adds cosmic shear data."""
    sacc_data = sacc.Sacc()

    z = (np.linspace(0, 4.0, 256) + 0.05).astype(np.float64)

    src_bins_centers = np.linspace(0.25, 0.75, 2)

    ccl_tracers = {}
    ccl_cell = {}
    cosmo = ccl.CosmologyVanillaLCDM()

    for i, mn in enumerate(src_bins_centers):
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.05**2)
        sacc_data.add_tracer("NZ", f"src{i}", z, dndz)
        ccl_tracers[f"src{i}"] = ccl.WeakLensingTracer(
            cosmo=cosmo,
            dndz=(z, dndz),
        )

    for i, j in upper_triangle_indices(len(src_bins_centers)):
        window = sacc.BandpowerWindow(window_function_data[1], window_function_data[0])
        Cell = ccl.angular_cl(
            cosmo=cosmo,
            tracer1=ccl_tracers[f"src{i}"],
            tracer2=ccl_tracers[f"src{j}"],
            ell=window_function_data[1],
        )
        Cell_binned = window_function_data[0].T @ Cell
        sacc_data.add_ell_cl(
            data_type="galaxy_shear_cl_ee",
            tracer1=f"src{i}",
            tracer2=f"src{j}",
            ell=window_function_data[2],
            x=Cell_binned,
            window=window,
        )
        ccl_cell[f"src{i}-src{j}"] = (
            Cell_binned,
            window_function_data[2],
            window_function_data[1],
            window_function_data[0],
        )

    sacc_data.add_covariance(np.identity(len(sacc_data)) * 0.01)

    return sacc_data, ccl_cell, cosmo


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
def fixture_cut(request):
    return request.param


two_point_yaml = """
correlation_space: harmonic
weak_lensing_factories:
  - type_source: default
    per_bin_systematics: []
    global_systematics: []
"""


def test_scale_cuts_with_bandpower_window_label(
    minimal_3x2pt_sacc, cut, tmp_path: Path
):
    """Test that the scale cuts with bandpower windows work correctly."""
    sacc_data, ccl_cell, cosmo = minimal_3x2pt_sacc

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
            cell if name != f"{cut[0]}-{cut[1]}" else cell[window_ell < cut[2]]
            for name, (cell, window_ell, _, _) in ccl_cell.items()
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
    minimal_3x2pt_sacc, cut, method, tmp_path: Path
):
    """Test that the scale cuts with bandpower windows work correctly."""
    sacc_data, ccl_cell, cosmo = minimal_3x2pt_sacc

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
                cell
                if name != f"{cut[0]}-{cut[1]}"
                else cell[window[ell < cut[2]].sum(axis=0) >= method[1]]
            )
            for name, (cell, _, ell, window) in ccl_cell.items()
        ]
    )
    # Check that the shapes match (i.e. that the correct data points are selected)
    assert ccl_cell_cut_data_vector.shape == likelihood.get_theory_vector().shape

    # Check that the computation of the theory vector matches the CCL computation
    assert np.allclose(likelihood.get_theory_vector(), ccl_cell_cut_data_vector)
