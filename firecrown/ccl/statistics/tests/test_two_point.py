import numpy as np
import pytest
import copy

import sacc
import pyccl as ccl

from ..two_point import TwoPointStatistic, _ell_for_xi, ELL_FOR_XI_DEFAULTS


class DummySource(object):
    pass


@pytest.mark.slow()
@pytest.mark.parametrize("ell_or_theta_max", [None, 80])
@pytest.mark.parametrize("ell_or_theta_min", [None, 20])
@pytest.mark.parametrize("ell_for_xi", [None, ELL_FOR_XI_DEFAULTS, {"mid": 100}])
@pytest.mark.parametrize("kind", ["cl", "gg", "gl", "l+", "l-"])
def test_two_point_sacc(kind, ell_for_xi, ell_or_theta_min, ell_or_theta_max, tmpdir):
    sacc_data = sacc.Sacc()

    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67,
    )

    sources = {}
    for i, mn in enumerate([0.25, 0.75]):
        sources["src%d" % i] = DummySource()
        z = np.linspace(0, 2, 50)
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)

        if ("g" in kind and i == 0) or kind == "gg":
            sources["src%d" % i].tracer_ = ccl.NumberCountsTracer(
                cosmo, has_rsd=False, dndz=(z, dndz), bias=(z, np.ones_like(z) * 2.0)
            )
        else:
            sources["src%d" % i].tracer_ = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz))

        sources["src%d" % i].sacc_tracer = "sacc_src%d" % i
        sources["src%d" % i].scale_ = i / 2.0 + 1.0
        sacc_data.add_tracer("NZ", "sacc_src%d" % i, z, dndz)

    # add extra data to make sure nothing weird is pulled back out
    sources["src3"] = DummySource()
    sources["src3"].tracer_ = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz))
    sources["src3"].sacc_tracer = "sacc_src3"
    sources["src3"].scale_ = 25.0
    sacc_data.add_tracer("NZ", "sacc_src3", z, dndz)
    sacc_data.add_tracer("NZ", "sacc_src5", z, dndz * 2)

    # compute the statistic
    tracers = [sources["src0"].tracer_, sources["src1"].tracer_]
    scale = np.prod([sources["src0"].scale_, sources["src1"].scale_])
    if kind == "cl":
        ell = np.logspace(1, 3, 10)
        ell_or_theta = ell
        cell = ccl.angular_cl(cosmo, *tracers, ell) * scale
        sacc_kind = "galaxy_shear_cl_ee"
        sacc_data.add_ell_cl(sacc_kind, "sacc_src0", "sacc_src1", ell, cell)
        sacc_data.add_ell_cl(sacc_kind, "sacc_src0", "sacc_src5", ell, cell * 2)
    else:
        theta = np.logspace(1, 2, 100)
        ell_or_theta = theta
        ell_for_xi_kws = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
        if ell_for_xi is not None:
            ell_for_xi_kws.update(ell_for_xi)
        ell = _ell_for_xi(**ell_for_xi_kws)
        cell = ccl.angular_cl(cosmo, *tracers, ell)
        xi = ccl.correlation(cosmo, ell, cell, theta / 60.0, corr_type=kind) * scale
        if kind == "gg":
            sacc_kind = "galaxy_density_xi"
        elif kind == "gl":
            sacc_kind = "galaxy_shearDensity_xi_t"
        elif kind == "l+":
            sacc_kind = "galaxy_shear_xi_plus"
        elif kind == "l-":
            sacc_kind = "galaxy_shear_xi_minus"
        sacc_data.add_theta_xi(sacc_kind, "sacc_src0", "sacc_src1", theta, xi)
        sacc_data.add_theta_xi(sacc_kind, "sacc_src0", "sacc_src5", theta, xi * 2)

    if ell_or_theta_min is not None:
        q = np.where(ell_or_theta >= ell_or_theta_min)
        ell_or_theta = ell_or_theta[q]
        if kind == "cl":
            cell = cell[q]
        else:
            xi = xi[q]

    if ell_or_theta_max is not None:
        q = np.where(ell_or_theta <= ell_or_theta_max)
        ell_or_theta = ell_or_theta[q]
        if kind == "cl":
            cell = cell[q]
        else:
            xi = xi[q]

    stat = TwoPointStatistic(
        sacc_data_type=sacc_kind,
        sources=["src0", "src1"],
        ell_for_xi=ell_for_xi,
        ell_or_theta_min=ell_or_theta_min,
        ell_or_theta_max=ell_or_theta_max,
    )
    stat.read(sacc_data, sources)
    stat.compute(cosmo, {}, sources, systematics=None)

    if ell_for_xi is not None:
        for key in ell_for_xi:
            assert ell_for_xi[key] == stat.ell_for_xi[key]

    assert np.array_equal(stat.ell_or_theta_, ell_or_theta)
    if ell_or_theta_min is not None:
        assert np.all(stat.ell_or_theta_ >= ell_or_theta_min)
    if ell_or_theta_max is not None:
        assert np.all(stat.ell_or_theta_ <= ell_or_theta_max)

    assert stat.ccl_kind == kind
    assert np.allclose(stat.scale_, np.prod(np.arange(2) / 2.0 + 1.0))
    if kind == "cl":
        assert np.allclose(stat.measured_statistic_, cell)
    else:
        assert np.allclose(stat.measured_statistic_, xi)

    assert np.allclose(stat.measured_statistic_, stat.predicted_statistic_)


@pytest.mark.slow()
@pytest.mark.parametrize("binning", ["log", "lin"])
@pytest.mark.parametrize("ell_or_theta_max", [None, 80])
@pytest.mark.parametrize("ell_or_theta_min", [None, 20])
@pytest.mark.parametrize("ell_for_xi", [None, ELL_FOR_XI_DEFAULTS, {"mid": 100}])
@pytest.mark.parametrize("kind", ["cl", "gg", "gl", "l+", "l-"])
def test_two_point_gen(
    kind, ell_for_xi, ell_or_theta_min, ell_or_theta_max, binning, tmpdir
):
    sacc_data = sacc.Sacc()

    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67,
    )

    sources = {}
    for i, mn in enumerate([0.25, 0.75]):
        sources["src%d" % i] = DummySource()
        z = np.linspace(0, 2, 50)
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)

        if ("g" in kind and i == 0) or kind == "gg":
            sources["src%d" % i].tracer_ = ccl.NumberCountsTracer(
                cosmo, has_rsd=False, dndz=(z, dndz), bias=(z, np.ones_like(z) * 2.0)
            )
        else:
            sources["src%d" % i].tracer_ = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz))

        sources["src%d" % i].sacc_tracer = "sacc_src%d" % i
        sources["src%d" % i].scale_ = i / 2.0 + 1.0
        sacc_data.add_tracer("NZ", "sacc_src%d" % i, z, dndz)

    # add extra data to make sure nothing weird is pulled back out
    sources["src3"] = DummySource()
    sources["src3"].tracer_ = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz))
    sources["src3"].sacc_tracer = "sacc_src3"
    sources["src3"].scale_ = 25.0
    sacc_data.add_tracer("NZ", "sacc_src3", z, dndz)
    sacc_data.add_tracer("NZ", "sacc_src5", z, dndz * 2)

    # compute the statistic
    tracers = [sources["src0"].tracer_, sources["src1"].tracer_]
    scale = np.prod([sources["src0"].scale_, sources["src1"].scale_])
    if kind == "cl":
        ell = np.logspace(1, 2, 10) if binning == "log" else np.linspace(10, 100, 10)
        ell = (
            np.sqrt(ell[1:] * ell[:-1])
            if binning == "log"
            else (ell[1:] + ell[:-1]) / 2
        )
        ell_or_theta = ell
        cell = ccl.angular_cl(cosmo, *tracers, ell) * scale
        sacc_kind = "galaxy_shear_cl_ee"
    else:
        theta = np.logspace(1, 2, 10) if binning == "log" else np.linspace(10, 100, 10)
        theta = (
            np.sqrt(theta[1:] * theta[:-1])
            if binning == "log"
            else (theta[1:] + theta[:-1]) / 2
        )
        ell_or_theta = theta
        ell_for_xi_kws = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
        if ell_for_xi is not None:
            ell_for_xi_kws.update(ell_for_xi)
        ell = _ell_for_xi(**ell_for_xi_kws)
        cell = ccl.angular_cl(cosmo, *tracers, ell)
        xi = ccl.correlation(cosmo, ell, cell, theta / 60.0, corr_type=kind) * scale
        if kind == "gg":
            sacc_kind = "galaxy_density_xi"
        elif kind == "gl":
            sacc_kind = "galaxy_shearDensity_xi_t"
        elif kind == "l+":
            sacc_kind = "galaxy_shear_xi_plus"
        elif kind == "l-":
            sacc_kind = "galaxy_shear_xi_minus"

    stat = TwoPointStatistic(
        sacc_data_type=sacc_kind,
        sources=["src0", "src1"],
        ell_for_xi=ell_for_xi,
        ell_or_theta_min=ell_or_theta_min,
        ell_or_theta_max=ell_or_theta_max,
        ell_or_theta={
            "min": 10,
            "max": 100,
            "n": 9,
            "binning": binning,
        },
    )

    if ell_or_theta_min is not None:
        q = np.where(ell_or_theta >= ell_or_theta_min)
        ell_or_theta = ell_or_theta[q]
        if kind == "cl":
            cell = cell[q]
        else:
            xi = xi[q]

    if ell_or_theta_max is not None:
        q = np.where(ell_or_theta <= ell_or_theta_max)
        ell_or_theta = ell_or_theta[q]
        if kind == "cl":
            cell = cell[q]
        else:
            xi = xi[q]

    stat.read(sacc_data, sources)
    stat.compute(cosmo, {}, sources, systematics=None)

    if ell_for_xi is not None:
        for key in ell_for_xi:
            assert ell_for_xi[key] == stat.ell_for_xi[key]

    assert np.array_equal(stat.ell_or_theta_, ell_or_theta)
    if ell_or_theta_min is not None:
        assert np.all(stat.ell_or_theta_ >= ell_or_theta_min)
    if ell_or_theta_max is not None:
        assert np.all(stat.ell_or_theta_ <= ell_or_theta_max)

    assert stat.ccl_kind == kind
    assert np.allclose(stat.scale_, np.prod(np.arange(2) / 2.0 + 1.0))
    if kind == "cl":
        assert np.allclose(stat.predicted_statistic_, cell)
    else:
        assert np.allclose(stat.predicted_statistic_, xi)

    assert np.allclose(stat.measured_statistic_, 0)


def test_two_point_raises_bad_sacc_data_type():
    with pytest.raises(ValueError) as e:
        TwoPointStatistic(sacc_data_type="blahXYZ", sources=["src0", "src1"])
    assert "blahXYZ" in str(e)


def test_two_point_raises_wrong_num_sources():
    with pytest.raises(ValueError) as e:
        TwoPointStatistic(sacc_data_type="galaxy_shear_cl_ee", sources=["src0"])
    assert "src0" in str(e)

@pytest.mark.filterwarnings("ignore:Empty index selected:UserWarning")
def test_two_point_raises_no_sacc_data():
    sacc_data = sacc.Sacc()

    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67,
    )

    sources = {}
    for i, mn in enumerate([0.25, 0.75]):
        sources["src%d" % i] = DummySource()
        z = np.linspace(0, 2, 50)
        dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)

        sources["src%d" % i].tracer_ = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz))

        sources["src%d" % i].sacc_tracer = "sacc_src%d" % i
        sources["src%d" % i].scale_ = i / 2.0 + 1.0
        sacc_data.add_tracer("NZ", "sacc_src%d" % i, z, dndz)

    # add extra data to make sure nothing weird is pulled back out
    sources["src3"] = DummySource()
    sources["src3"].tracer_ = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz))
    sources["src3"].sacc_tracer = "sacc_src3"
    sources["src3"].scale_ = 25.0
    sacc_data.add_tracer("NZ", "sacc_src3", z, dndz)
    sacc_data.add_tracer("NZ", "sacc_src5", z, dndz * 2)

    # compute the statistic
    tracers = [sources["src0"].tracer_, sources["src1"].tracer_]
    scale = np.prod([sources["src0"].scale_, sources["src1"].scale_])
    ell = np.logspace(1, 3, 10)
    cell = ccl.angular_cl(cosmo, *tracers, ell) * scale
    sacc_kind = "galaxy_shear_cl_ee"
    sacc_data.add_ell_cl(sacc_kind, "sacc_src0", "sacc_src1", ell, cell)
    sacc_data.add_ell_cl(sacc_kind, "sacc_src0", "sacc_src5", ell, cell * 2)

    stat = TwoPointStatistic(sacc_data_type=sacc_kind, sources=["src0", "src3"])
    with pytest.raises(RuntimeError) as e:
        stat.read(sacc_data, sources)
    assert "have no 2pt data" in str(e)
