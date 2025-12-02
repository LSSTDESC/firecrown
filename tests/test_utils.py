"""
Tests for the firecrown.utils modle.
"""

import pytest
import numpy as np
import pyccl
from numpy.testing import assert_allclose

from firecrown.utils import (
    upper_triangle_indices,
    save_to_sacc,
    compare_optional_arrays,
    compare_optionals,
    base_model_from_yaml,
    base_model_to_yaml,
    ClIntegrationMethod,
    ClLimberMethod,
    ClIntegrationOptions,
    make_log_interpolator,
    cached_angular_cl,
)


def test_upper_triangle_indices_nonzero():
    indices = list(upper_triangle_indices(3))
    assert indices == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_upper_triangle_indices_zero():
    indices = list(upper_triangle_indices(0))
    assert not indices


def test_save_to_sacc(trivial_stats, sacc_data_for_trivial_stat):
    stat = trivial_stats[0]
    stat.read(sacc_data_for_trivial_stat)
    idx = np.arange(stat.count)
    new_data_vector = 3 * stat.get_data_vector()[idx]

    new_sacc = save_to_sacc(
        sacc_data=sacc_data_for_trivial_stat,
        data_vector=new_data_vector,
        indices=idx,
        strict=True,
    )
    assert all(new_sacc.data[i].value == d for i, d in zip(idx, new_data_vector))


def test_save_to_sacc_strict_fail(trivial_stats, sacc_data_for_trivial_stat):
    stat = trivial_stats[0]
    stat.read(sacc_data_for_trivial_stat)
    idx = np.arange(stat.count - 1)
    new_data_vector = stat.get_data_vector()[idx]

    with pytest.raises(RuntimeError):
        _ = save_to_sacc(
            sacc_data=sacc_data_for_trivial_stat,
            data_vector=new_data_vector,
            indices=idx,
            strict=True,
        )


def test_save_to_sacc_non_sttrict(trivial_stats, sacc_data_for_trivial_stat):
    stat = trivial_stats[0]
    stat.read(sacc_data_for_trivial_stat)
    idx = np.arange(stat.count)
    new_data_vector = 3 * stat.get_data_vector()[idx]

    new_sacc = save_to_sacc(
        sacc_data=sacc_data_for_trivial_stat,
        data_vector=new_data_vector,
        indices=idx,
        strict=False,
    )
    assert all(new_sacc.data[i].value == d for i, d in zip(idx, new_data_vector))


def test_compare_optional_arrays_():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    assert compare_optional_arrays(x, y)

    z = np.array([1, 2, 4])
    assert not compare_optional_arrays(x, z)

    a = None
    b = None
    assert compare_optional_arrays(a, b)

    q = np.array([1, 2, 3])
    assert not compare_optional_arrays(q, a)

    assert not compare_optional_arrays(a, q)


def test_base_model_from_yaml_wrong():
    with pytest.raises(ValueError):
        _ = base_model_from_yaml(str, "wrong")


def test_compare_optionals():
    x = "test"
    y = "test"
    assert compare_optionals(x, y)

    z = "test2"
    assert not compare_optionals(x, z)

    a = None
    b = None
    assert compare_optionals(a, b)

    q = np.array([1, 2, 3])
    assert not compare_optionals(q, a)

    assert not compare_optionals(a, q)


@pytest.fixture(
    name="limber_method",
    params=[ClLimberMethod.GSL_SPLINE, ClLimberMethod.GSL_QAG_QUAD],
)
def fixture_limber_method(request):
    return request.param


@pytest.fixture(
    name="limber_max_error",
    params=[None, 0.1, 0.2],
)
def fixture_limber_max_error(request):
    return request.param


@pytest.fixture(name="fkem_chi_min", params=[None, 0.05, 0.082])
def fixture_fkem_chi_min(request):
    return request.param


@pytest.fixture(name="fkem_Nchi", params=[None, 10, 20])
def fixture_fkem_Nchi(request):
    return request.param


@pytest.fixture(
    name="l_limber",
    params=[40, 60],
)
def fixture_l_limber(request):
    return request.param


def test_cl_integration_options_limber(limber_method: ClLimberMethod):
    int_options = ClIntegrationOptions(
        method=ClIntegrationMethod.LIMBER, limber_method=limber_method
    )
    assert int_options.limber_method == limber_method
    assert int_options.method == ClIntegrationMethod.LIMBER

    args = int_options.get_angular_cl_args()
    assert args["limber_integration_method"] == limber_method.lower().removeprefix(
        "gsl_"
    )
    assert args["l_limber"] == -1
    assert len(args) == 2

    int_options_yaml = base_model_to_yaml(int_options)
    int_options2 = base_model_from_yaml(ClIntegrationOptions, int_options_yaml)
    assert int_options == int_options2


def test_cl_integration_options_limber_yaml(limber_method: ClLimberMethod):
    int_options_yaml = f"""
    method: LIMBER
    limber_method: {limber_method.lower()}
    """
    int_options = base_model_from_yaml(ClIntegrationOptions, int_options_yaml)
    assert int_options.limber_method == limber_method
    assert int_options.method == ClIntegrationMethod.LIMBER


def test_cl_integration_options_fkem_auto(
    limber_method: ClLimberMethod,
    limber_max_error: float | None,
    fkem_chi_min: float | None,
    fkem_Nchi: int | None,
):
    int_options = ClIntegrationOptions(
        method=ClIntegrationMethod.FKEM_AUTO,
        limber_method=limber_method,
        limber_max_error=limber_max_error,
        fkem_chi_min=fkem_chi_min,
        fkem_Nchi=fkem_Nchi,
    )
    assert int_options.limber_method == limber_method
    assert int_options.method == ClIntegrationMethod.FKEM_AUTO

    args = int_options.get_angular_cl_args()
    assert args["limber_integration_method"] == limber_method.lower().removeprefix(
        "gsl_"
    )
    assert args["l_limber"] == "auto"
    assert args["non_limber_integration_method"] == "FKEM"
    expected_len = 3
    if limber_max_error is not None:
        assert args["limber_max_error"] == limber_max_error
        expected_len += 1

    if fkem_chi_min is not None:
        assert args["fkem_chi_min"] == fkem_chi_min
        expected_len += 1

    if fkem_Nchi is not None:
        assert args["fkem_Nchi"] == fkem_Nchi
        expected_len += 1

    assert len(args) == expected_len

    int_options_yaml = base_model_to_yaml(int_options)
    int_options2 = base_model_from_yaml(ClIntegrationOptions, int_options_yaml)
    assert int_options == int_options2


def test_cl_integration_options_fkem_auto_yaml(
    limber_method: ClLimberMethod,
    limber_max_error: float | None,
    fkem_chi_min: float | None,
    fkem_Nchi: int | None,
):
    int_options_yaml = f"""
    method: FKEM_AUTO
    limber_method: {limber_method.lower()}
    limber_max_error: {limber_max_error if limber_max_error is not None else "null"}
    fkem_chi_min: {fkem_chi_min if fkem_chi_min is not None else "null"}
    fkem_Nchi: {fkem_Nchi if fkem_Nchi is not None else "null"}
    """
    int_options = base_model_from_yaml(ClIntegrationOptions, int_options_yaml)
    assert int_options.limber_method == limber_method
    assert int_options.method == ClIntegrationMethod.FKEM_AUTO
    assert int_options.limber_max_error == limber_max_error
    assert int_options.fkem_chi_min == fkem_chi_min
    assert int_options.fkem_Nchi == fkem_Nchi


def test_cl_integration_options_fkem_l_limber(
    limber_method: ClLimberMethod,
    fkem_chi_min: float | None,
    fkem_Nchi: int | None,
    l_limber: int,
):
    int_options = ClIntegrationOptions(
        method=ClIntegrationMethod.FKEM_L_LIMBER,
        limber_method=limber_method,
        l_limber=l_limber,
        fkem_chi_min=fkem_chi_min,
        fkem_Nchi=fkem_Nchi,
    )
    assert int_options.limber_method == limber_method
    assert int_options.method == ClIntegrationMethod.FKEM_L_LIMBER

    args = int_options.get_angular_cl_args()
    assert args["limber_integration_method"] == limber_method.lower().removeprefix(
        "gsl_"
    )
    assert args["l_limber"] == int_options.l_limber
    assert args["non_limber_integration_method"] == "FKEM"
    expected_len = 3
    if fkem_chi_min is not None:
        assert args["fkem_chi_min"] == fkem_chi_min
        expected_len += 1

    if fkem_Nchi is not None:
        assert args["fkem_Nchi"] == fkem_Nchi
        expected_len += 1

    assert len(args) == expected_len

    int_options_yaml = base_model_to_yaml(int_options)
    int_options2 = base_model_from_yaml(ClIntegrationOptions, int_options_yaml)
    assert int_options == int_options2


def test_cl_integration_options_fkem_l_limber_yaml(
    limber_method: ClLimberMethod,
    fkem_chi_min: float | None,
    fkem_Nchi: int | None,
    l_limber: int,
):
    int_options_yaml = f"""
    method: FKEM_L_LIMBER
    limber_method: {limber_method.lower()}
    l_limber: {l_limber}
    fkem_chi_min: {fkem_chi_min if fkem_chi_min is not None else "null"}
    fkem_Nchi: {fkem_Nchi if fkem_Nchi is not None else "null"}
    """
    int_options = base_model_from_yaml(ClIntegrationOptions, int_options_yaml)
    assert int_options.limber_method == limber_method
    assert int_options.method == ClIntegrationMethod.FKEM_L_LIMBER
    assert int_options.l_limber == l_limber
    assert int_options.fkem_chi_min == fkem_chi_min
    assert int_options.fkem_Nchi == fkem_Nchi


def test_cl_integration_options_yaml_invalid():
    int_options_yaml = """
    method: Im_not_a_valid_method
    limber_method: qag_quad
    limber_max_error: 0.1
    fkem_chi_min: 0.1
    fkem_Nchi: 10
    """
    with pytest.raises(
        ValueError,
        match=("Invalid value for ClIntegrationMethod: Im_not_a_valid_method"),
    ):
        base_model_from_yaml(ClIntegrationOptions, int_options_yaml)

    int_options_yaml = """
    method: 0.34
    limber_method: qag_quad
    limber_max_error: 0.1
    fkem_chi_min: 0.1
    fkem_Nchi: 10
    """
    with pytest.raises(
        ValueError,
        match=("Input should be an instance of ClIntegrationMethod"),
    ):
        base_model_from_yaml(ClIntegrationOptions, int_options_yaml)

    int_options_yaml = """
    method: limber
    limber_method: Im_not_a_valid_limber_method
    limber_max_error: 0.1
    fkem_chi_min: 0.1
    fkem_Nchi: 10
    """
    with pytest.raises(
        ValueError,
        match=("Invalid value for ClLimberMethod: Im_not_a_valid_limber_method"),
    ):
        base_model_from_yaml(ClIntegrationOptions, int_options_yaml)

    int_options_yaml = """
    method: limber
    limber_method: 123
    limber_max_error: 0.1
    fkem_chi_min: 0.1
    fkem_Nchi: 10
    """
    with pytest.raises(
        ValueError,
        match=("Input should be an instance of ClLimberMethod"),
    ):
        base_model_from_yaml(ClIntegrationOptions, int_options_yaml)


def test_cl_integration_options_limber_invalid():
    with pytest.raises(
        ValueError, match="l_limber is incompatible with ClIntegrationMethod.LIMBER"
    ):
        ClIntegrationOptions(
            method=ClIntegrationMethod.LIMBER,
            limber_method=ClLimberMethod.GSL_QAG_QUAD,
            l_limber=3,
        )
    with pytest.raises(
        ValueError,
        match="limber_max_error is incompatible with ClIntegrationMethod.LIMBER",
    ):
        ClIntegrationOptions(
            method=ClIntegrationMethod.LIMBER,
            limber_method=ClLimberMethod.GSL_QAG_QUAD,
            limber_max_error=0.1,
        )

    with pytest.raises(
        ValueError,
        match="fkem_chi_min is incompatible with ClIntegrationMethod.LIMBER",
    ):
        ClIntegrationOptions(
            method=ClIntegrationMethod.LIMBER,
            limber_method=ClLimberMethod.GSL_QAG_QUAD,
            fkem_chi_min=0.1,
        )
    with pytest.raises(
        ValueError,
        match="fkem_Nchi is incompatible with ClIntegrationMethod.LIMBER",
    ):
        ClIntegrationOptions(
            method=ClIntegrationMethod.LIMBER,
            limber_method=ClLimberMethod.GSL_QAG_QUAD,
            fkem_Nchi=3,
        )


def test_cl_integration_options_fkem_auto_invalid():
    with pytest.raises(
        ValueError,
        match="l_limber is incompatible with ClIntegrationMethod.FKEM_AUTO",
    ):
        ClIntegrationOptions(
            method=ClIntegrationMethod.FKEM_AUTO,
            limber_method=ClLimberMethod.GSL_QAG_QUAD,
            l_limber=3,
        )


def test_cl_integration_options_fkem_l_limber_invalid():
    with pytest.raises(
        ValueError,
        match="limber_max_error is incompatible with ClIntegrationMethod.FKEM_L_LIMBER",
    ):
        ClIntegrationOptions(
            method=ClIntegrationMethod.FKEM_L_LIMBER,
            limber_method=ClLimberMethod.GSL_QAG_QUAD,
            limber_max_error=0.1,
            l_limber=3,
        )

    with pytest.raises(
        ValueError,
        match="l_limber must be set for FKEM_L_LIMBER",
    ):
        ClIntegrationOptions(
            method=ClIntegrationMethod.FKEM_L_LIMBER,
            limber_method=ClLimberMethod.GSL_QAG_QUAD,
        )


def test_cl_integration_test_cmparison():
    int_options = ClIntegrationOptions(
        method=ClIntegrationMethod.LIMBER, limber_method=ClLimberMethod.GSL_SPLINE
    )
    assert int_options == ClIntegrationOptions(
        method=ClIntegrationMethod.LIMBER, limber_method=ClLimberMethod.GSL_SPLINE
    )
    assert int_options != ClIntegrationOptions(
        method=ClIntegrationMethod.FKEM_AUTO, limber_method=ClLimberMethod.GSL_SPLINE
    )


def test_make_log_interpolator_positive():
    x = np.arange(1, 3000)
    y = np.exp(x / 1.0e3) * x**1.43

    f = make_log_interpolator(x, y)

    assert_allclose(f(x), y, atol=0.0, rtol=1e-12)


def test_make_log_interpolator_negative():
    x = np.arange(1, 3000)
    y = np.exp(x / 1.0e3) * (x - 1000.5)

    f = make_log_interpolator(x, y)

    assert_allclose(f(x), y, atol=0.0, rtol=1e-12)


def test_cached_angular_cl(tools_with_vanilla_cosmology):
    tools = tools_with_vanilla_cosmology
    cosmo = tools.get_ccl_cosmology()

    z = np.linspace(0, 1, 100)
    nz = np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2)
    tracer1 = pyccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    tracer2 = pyccl.WeakLensingTracer(cosmo, dndz=(z, nz))

    ells = tuple([10, 20, 30])

    cl = cached_angular_cl(
        cosmo, (tracer1, tracer2), ells, p_of_k_a="delta_matter:delta_matter"
    )
    assert len(cl) == len(ells)
    assert np.all(np.isfinite(cl))


def test_cached_angular_cl_with_options(tools_with_vanilla_cosmology):
    tools = tools_with_vanilla_cosmology
    cosmo = tools.get_ccl_cosmology()

    z = np.linspace(0, 1, 100)
    nz = np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2)
    tracer1 = pyccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    tracer2 = pyccl.WeakLensingTracer(cosmo, dndz=(z, nz))

    ells = tuple([10, 20, 30])
    int_options = ClIntegrationOptions(
        method=ClIntegrationMethod.LIMBER, limber_method=ClLimberMethod.GSL_SPLINE
    )

    cl = cached_angular_cl(
        cosmo,
        (tracer1, tracer2),
        ells,
        p_of_k_a="delta_matter:delta_matter",
        int_options=int_options,
    )
    assert len(cl) == len(ells)
    assert np.all(np.isfinite(cl))


@pytest.mark.parametrize(
    "method,expected",
    [
        (ClIntegrationMethod.LIMBER, "limber"),
        (ClIntegrationMethod.FKEM_AUTO, "fkem_auto"),
        (ClIntegrationMethod.FKEM_L_LIMBER, "fkem_l_limber"),
    ],
)
def test_yaml_serializable_to_yaml(method, expected):
    # Add test to verify all enum values are covered
    all_methods = set(ClIntegrationMethod)
    test_methods = {
        param[0] for param in test_yaml_serializable_to_yaml.pytestmark[0].args[1]
    }
    assert (
        all_methods == test_methods
    ), f"Test missing enum values: {all_methods - test_methods}"

    yaml_str = method.to_yaml()
    assert expected in yaml_str.lower()


@pytest.mark.parametrize(
    "yaml_str,expected",
    [
        ("limber", ClIntegrationMethod.LIMBER),
        ("fkem_auto", ClIntegrationMethod.FKEM_AUTO),
        ("fkem_l_limber", ClIntegrationMethod.FKEM_L_LIMBER),
    ],
)
def test_yaml_serializable_from_yaml(yaml_str, expected):
    # Add test to verify all enum values are covered
    all_methods = set(ClIntegrationMethod)
    test_methods = {
        param[1] for param in test_yaml_serializable_from_yaml.pytestmark[0].args[1]
    }
    assert (
        all_methods == test_methods
    ), f"Test missing enum values: {all_methods - test_methods}"

    restored = ClIntegrationMethod.from_yaml(yaml_str)
    assert restored == expected
