"""Tests for window functions.

These tests are based on the bug report at github.com/LSSTDESC/firecrown/issues/398.
They were added as regression tests to ensure that the Firecrown codebase remains stable
and consistent.

The variables SRC2_SRC2_CL_VANILLA_LCDM, LENS0_SRC2_CL_VANILLA_LCDM, and
LENS0_LENS0_CL_VANILLA_LCDM are the expected values of the correlation functions for the
src2_src2, lens0_src2, and lens0_lens0 pairs, respectively. These values were computed
using the vanilla LCDM cosmology. They will require updating if CCL or the cosmology
changes.

"""

import os
import sacc

from numpy.testing import assert_allclose

from firecrown.updatable import get_default_params_map
import firecrown.likelihood._weak_lensing as wl
from firecrown.likelihood._two_point import TwoPoint
from firecrown.likelihood._gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood._likelihood import Likelihood, NamedParameters
from firecrown.ccl_factory import CCLFactory, PoweSpecAmplitudeParameter


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

    modeling_tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    likelihood = ConstGaussian(statistics=[src2_src2, lens0_lens0, lens0_src2])

    likelihood.read(sacc_data)

    return likelihood, modeling_tools


SACC_FILE = f"{os.environ.get('FIRECROWN_DIR')}/tests/bug_398.sacc.gz"

SRC2_SRC2_CL_VANILLA_LCDM = [
    1.6996796409559522e-08,
    1.3830498244567276e-08,
    1.1171074087134096e-08,
    8.71397622511587e-09,
    6.685475025309092e-09,
    5.090338072385348e-09,
    3.7790146627976595e-09,
    2.7640695818270494e-09,
    2.0406629675375553e-09,
    1.5072643669938013e-09,
    1.121245228055608e-09,
    8.4630157854997e-10,
    6.428900343759968e-10,
    4.921750354911857e-10,
    3.781418298068616e-10,
    2.89146868380689e-10,
    2.1888336077880626e-10,
    1.6323440308029163e-10,
    1.1932915993983155e-10,
    8.522909391801528e-11,
]

LENS0_SRC2_CL_VANILLA_LCDM = [
    7.0323092453773845e-09,
    5.494579422459329e-09,
    4.313847024362691e-09,
    3.260791730838365e-09,
    2.4107696268452148e-09,
    1.7854547756646923e-09,
    1.3207859855326834e-09,
    9.797352749677378e-10,
    7.363315698670614e-10,
    5.58205206395644e-10,
    4.2747485171361307e-10,
    3.306254815208201e-10,
    2.552297716630808e-10,
    1.9619040977545633e-10,
    1.4938778467536056e-10,
    1.1182312905569403e-10,
    8.199329491647421e-11,
    5.873699556722951e-11,
    4.100276444412095e-11,
    2.78742168121908e-11,
]

LENS0_LENS0_CL_VANILLA_LCDM = [
    4.377762702278542e-09,
    3.3760971439877716e-09,
    2.6144633981917815e-09,
    1.94567063609179e-09,
    1.4272571514306968e-09,
    1.061358584050687e-09,
    7.911318322522331e-10,
    5.922011374174115e-10,
    4.5068095158916795e-10,
    3.4553695338497153e-10,
    2.667806830617964e-10,
    2.0696427103449728e-10,
    1.5934177750242695e-10,
    1.2145642020574242e-10,
    9.12602765151705e-11,
    6.713910843193902e-11,
    4.8245792476518054e-11,
    3.3819274168976346e-11,
    2.3093491365788615e-11,
    1.5369307085611505e-11,
]


def test_broken_window_function() -> None:
    likelihood, modeling_tools = build_likelihood(
        build_parameters=NamedParameters({"sacc_data": SACC_FILE})
    )
    assert likelihood is not None
    assert modeling_tools is not None


def test_eval_cl_window_src2_src2() -> None:
    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

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


def test_eval_cl_window_lens0_src2() -> None:
    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

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


def test_eval_cl_window_lens0_lens0() -> None:
    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

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


def test_compute_likelihood_src0_src0() -> None:
    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )
    params = get_default_params_map(tools)
    tools.update(params)
    tools.prepare()

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
