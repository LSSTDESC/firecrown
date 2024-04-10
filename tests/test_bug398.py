"""Tests for window functions."""

import os
import sacc

import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.likelihood import Likelihood


def build_likelihood(build_parameters) -> tuple[Likelihood, ModelingTools]:
    """Sample build_likelihood function for this test."""
    # Load sacc file
    sacc_data = build_parameters["sacc_data"]
    if isinstance(sacc_data, str):
        sacc_data = sacc.Sacc.load_fits(sacc_data)

    tracer = wl.WeakLensing(sacc_tracer="src0")

    stat = TwoPoint(
        source0=tracer,
        source1=tracer,
        sacc_data_type="galaxy_shear_cl_ee",
    )

    modeling_tools = ModelingTools()
    likelihood = ConstGaussian(statistics=[stat])

    likelihood.read(sacc_data)

    return likelihood, modeling_tools


SACC_FILE = f"{os.environ.get('FIRECROWN_DIR')}/tests/bug_398.sacc.gz"


def test_broken_window_function():
    likelihood, modeling_tools = build_likelihood(
        build_parameters={"sacc_data": SACC_FILE}
    )
    assert likelihood is not None
    assert modeling_tools is not None


if __name__ == "__main__":
    test_broken_window_function()
