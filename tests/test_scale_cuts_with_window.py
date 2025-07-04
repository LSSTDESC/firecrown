"""Test that the scale cuts with bandpower windows work correctly."""

import tempfile
import pytest
import numpy as np

import pyccl as ccl

import sacc

from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.factories import (
    DataSourceSacc,
    TwoPointExperiment,
    TwoPointFactory,
)
from firecrown.data_functions import (
    TwoPointBinFilterCollection,
    TwoPointBinFilter,
)
from firecrown.metadata_types import Galaxies
from firecrown.parameters import ParamsMap

from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import get_default_params_map
from firecrown.utils import base_model_from_yaml, upper_triangle_indices


@pytest.fixture(name="minimal_3x2pt_sacc")
def fixture_minimal_3x2pt_sacc():
    """Generate SACC data with window functions.
    At the moment, this only adds cosmic shear data."""
    sacc_data = sacc.Sacc()

    z = (np.linspace(0, 4.0, 256) + 0.05).astype(np.float64)
    n_ell_bin = 5
    ell_bin_edges = np.geomspace(10, 500, n_ell_bin + 1)
    ell_bin_centers = np.sqrt(ell_bin_edges[:-1] * ell_bin_edges[1:])
    ell_bin_widths = np.diff(ell_bin_edges)

    ell_window = np.arange(800)
    window_function = np.zeros((ell_window.shape[0], n_ell_bin))
    for i, (mu_ell, sigma_ell) in enumerate(zip(ell_bin_centers, ell_bin_widths)):
        w_col = np.exp(-0.5 * (ell_window - mu_ell) ** 2 / (sigma_ell) ** 2)
        threshold = np.max(w_col) * (1 - np.finfo(float).eps)
        w_col[w_col < threshold] = 0.0
        window_function[:, i] = w_col

    window_function = window_function / window_function.sum(axis=0, keepdims=True)

    n_source = 2
    src_bins_centers = np.linspace(0.25, 0.75, n_source)

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
        window = sacc.BandpowerWindow(ell_window, window_function)
        Cell = ccl.angular_cl(
            cosmo=cosmo,
            tracer1=ccl_tracers[f"src{i}"],
            tracer2=ccl_tracers[f"src{j}"],
            ell=ell_window,
        )
        Cell_binned = window_function.T @ Cell
        sacc_data.add_ell_cl(
            data_type="galaxy_shear_cl_ee",
            tracer1=f"src{i}",
            tracer2=f"src{j}",
            ell=ell_bin_centers,
            x=Cell_binned,
            window=window,
        )
        ccl_cell[f"src{i}-src{j}"] = (Cell_binned, ell_bin_centers)

    sacc_data.add_covariance(np.identity(len(sacc_data)) * 0.01)

    return sacc_data, ccl_cell, cosmo


two_point_yaml = """
correlation_space: harmonic
weak_lensing_factories:
  - type_source: default
    per_bin_systematics: []
    global_systematics: []
"""


def test_scale_cuts_with_bandpower_window(minimal_3x2pt_sacc):
    """Test that the scale cuts with bandpower windows work correctly."""
    sacc_data, ccl_cell, cosmo = minimal_3x2pt_sacc

    cut = ("src0", "src0", 100)

    tp_factory = base_model_from_yaml(TwoPointFactory, two_point_yaml)

    with tempfile.NamedTemporaryFile(
        suffix=".sacc", delete=True, delete_on_close=False
    ) as tmp_file:
        sacc_data.save_fits(tmp_file.name, overwrite=True)

        two_point_experiment = TwoPointExperiment(
            two_point_factory=tp_factory,
            data_source=DataSourceSacc(
                sacc_data_file=tmp_file.name,
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
                        )
                    ],
                ),
            ),
        )
        likelihood = two_point_experiment.make_likelihood()

    tools = ModelingTools(
        ccl_factory=two_point_experiment.ccl_factory,
    )
    params = get_default_params_map(tools, likelihood)
    params.update(
        ParamsMap({k: v for k, v in cosmo.to_dict().items() if isinstance(v, float)})
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
            cell if name != f"{cut[0]}-{cut[1]}" else cell[ell < cut[2]]
            for name, (cell, ell) in ccl_cell.items()
        ]
    )
    # Check that the shapes match (i.e. that the correct data points are selected)
    assert ccl_cell_cut_data_vector.shape == likelihood.get_theory_vector().shape

    # Check that the computation of the theory vector matches the CCL computation
    assert np.allclose(likelihood.get_theory_vector(), ccl_cell_cut_data_vector)
