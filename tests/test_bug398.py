"""Tests for window functions."""

import os
import sacc
import pyccl

from numpy.testing import assert_allclose

import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.parameters import ParamsMap


def build_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """Sample build_likelihood function for this test."""
    # Load sacc file
    sacc_data = build_parameters.get_string("sacc_data")
    if isinstance(sacc_data, str):
        sacc_data = sacc.Sacc.load_fits(sacc_data)

    src2 = wl.WeakLensing(sacc_tracer="src2")
    lens0 = wl.WeakLensing(sacc_tracer="lens0")

    src2_src2 = TwoPoint(
        source0=src2,
        source1=src2,
        sacc_data_type="galaxy_shear_cl_ee",
    )
    lens0_src2 = TwoPoint(
        source0=lens0,
        source1=src2,
        sacc_data_type="galaxy_shearDensity_cl_e",
    )
    lens0_lens0 = TwoPoint(
        source0=lens0,
        source1=lens0,
        sacc_data_type="galaxy_density_cl",
    )

    modeling_tools = ModelingTools()
    likelihood = ConstGaussian(statistics=[src2_src2, lens0_lens0, lens0_src2])

    likelihood.read(sacc_data)

    return likelihood, modeling_tools


SACC_FILE = f"{os.environ.get('FIRECROWN_DIR')}/tests/bug_398.sacc.gz"
SRC2_SRC2_CL_VANILLA_LCDM = [
    1.69967963e-08,
    1.38304980e-08,
    1.11710737e-08,
    8.71397605e-09,
    6.68547501e-09,
    5.09033805e-09,
    3.77901452e-09,
    2.76406946e-09,
    2.04066288e-09,
    1.50726429e-09,
    1.12124517e-09,
    8.46301538e-10,
    6.42890014e-10,
    4.92175018e-10,
    3.78141820e-10,
    2.89146863e-10,
    2.18883359e-10,
    1.63234403e-10,
    1.19329161e-10,
    8.52290961e-11,
]
LENS0_SRC2_CL_VANILLA_LCDM = [
    7.03230915e-09,
    5.49457937e-09,
    4.31384700e-09,
    3.26079168e-09,
    2.41076958e-09,
    1.78545473e-09,
    1.32078595e-09,
    9.79735232e-10,
    7.36331539e-10,
    5.58205184e-10,
    4.27474837e-10,
    3.30625472e-10,
    2.55229767e-10,
    1.96190406e-10,
    1.49387782e-10,
    1.11823128e-10,
    8.19932945e-11,
    5.87369958e-11,
    4.10027651e-11,
    2.78742176e-11,
]
LENS0_LENS0_CL_VANILLA_LCDM = [
    4.37776266e-09,
    3.37609715e-09,
    2.61446339e-09,
    1.94567058e-09,
    1.42725712e-09,
    1.06135857e-09,
    7.91131823e-10,
    5.92201116e-10,
    4.50680940e-10,
    3.45536945e-10,
    2.66780679e-10,
    2.06964268e-10,
    1.59341776e-10,
    1.21456419e-10,
    9.12602758e-11,
    6.71391079e-11,
    4.82457923e-11,
    3.38192742e-11,
    2.30934916e-11,
    1.53693074e-11,
]


def test_broken_window_function():
    likelihood, modeling_tools = build_likelihood(
        build_parameters=NamedParameters({"sacc_data": SACC_FILE})
    )
    assert likelihood is not None
    assert modeling_tools is not None


def test_eval_cl_window_src2_src2():
    tools = ModelingTools()
    cosmo = pyccl.CosmologyVanillaLCDM()
    params = ParamsMap()

    tools.update(params)
    tools.prepare(cosmo)

    sacc_data = sacc.Sacc.load_fits(SACC_FILE)
    src2 = wl.WeakLensing(sacc_tracer="src2")

    src2_src2 = TwoPoint(
        source0=src2,
        source1=src2,
        sacc_data_type="galaxy_shear_cl_ee",
    )

    src2_src2.read(sacc_data)
    src2_src2.update(params)

    theory_vector = src2_src2.compute_theory_vector(tools)
    assert_allclose(theory_vector, SRC2_SRC2_CL_VANILLA_LCDM, rtol=1e-8)


def test_eval_cl_window_lens0_src2():
    tools = ModelingTools()
    cosmo = pyccl.CosmologyVanillaLCDM()
    params = ParamsMap()

    tools.update(params)
    tools.prepare(cosmo)

    sacc_data = sacc.Sacc.load_fits(SACC_FILE)
    src2 = wl.WeakLensing(sacc_tracer="src2")
    lens0 = wl.WeakLensing(sacc_tracer="lens0")

    lens0_src2 = TwoPoint(
        source0=lens0,
        source1=src2,
        sacc_data_type="galaxy_shearDensity_cl_e",
    )

    lens0_src2.read(sacc_data)
    lens0_src2.update(params)

    theory_vector = lens0_src2.compute_theory_vector(tools)
    assert_allclose(theory_vector, LENS0_SRC2_CL_VANILLA_LCDM, rtol=1e-8)


def test_eval_cl_window_lens0_lens0():
    tools = ModelingTools()
    cosmo = pyccl.CosmologyVanillaLCDM()
    params = ParamsMap()

    tools.update(params)
    tools.prepare(cosmo)

    sacc_data = sacc.Sacc.load_fits(SACC_FILE)
    lens0 = wl.WeakLensing(sacc_tracer="lens0")

    lens0_lens0 = TwoPoint(
        source0=lens0,
        source1=lens0,
        sacc_data_type="galaxy_density_cl",
    )

    lens0_lens0.read(sacc_data)
    lens0_lens0.update(params)

    theory_vector = lens0_lens0.compute_theory_vector(tools)
    assert_allclose(theory_vector, LENS0_LENS0_CL_VANILLA_LCDM, rtol=1e-8)


def test_compute_likelihood_src0_src0():
    tools = ModelingTools()
    cosmo = pyccl.CosmologyVanillaLCDM()
    params = ParamsMap()

    tools.update(params)
    tools.prepare(cosmo)

    sacc_data = sacc.Sacc.load_fits(SACC_FILE)
    src0 = wl.WeakLensing(sacc_tracer="src0")

    src0_src0 = TwoPoint(
        source0=src0,
        source1=src0,
        sacc_data_type="galaxy_shear_cl_ee",
    )

    likelihood = ConstGaussian(statistics=[src0_src0])
    likelihood.read(sacc_data)
    likelihood.update(params)

    log_like = likelihood.compute_loglike(tools)
    # Compare to 73af686f050ea60d349a0e6792a4832f2e7f554e
    assert_allclose(log_like, -15.014821, rtol=1e-7)


if __name__ == "__main__":
    test_broken_window_function()
    test_eval_cl_window_src2_src2()
    test_eval_cl_window_lens0_src2()
    test_eval_cl_window_lens0_lens0()
    test_compute_likelihood_src0_src0()
