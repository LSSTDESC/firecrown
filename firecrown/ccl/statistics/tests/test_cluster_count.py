import numpy as np
from scipy.interpolate import Akima1DInterpolator

import sacc
import pyccl as ccl

from ..cluster_count import ClusterCountStatistic


class DummySource(object):
    pass


def test_cluster_count_sacc(tmpdir):
    sacc_data = sacc.Sacc()

    params = dict(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67,
    )
    cosmo = ccl.Cosmology(**params)

    mn = 0.5
    z = np.linspace(0, 2, 50)
    dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)
    nrm = np.max(dndz)
    dndz /= nrm
    sacc_data.add_tracer(
        "NZ",
        "trc1",
        z,
        dndz,
        metadata={
            "lnlam_min": 14,
            "lnlam_max": 16,
            "area_sd": 15.1 * (180.0 / np.pi) ** 2,
        },
    )

    intp = Akima1DInterpolator(z, dndz)

    def _sel(m, a):
        a = np.atleast_1d(a)
        m = np.atleast_1d(m)
        z = 1.0 / a - 1.0
        logm = np.log10(m)
        zsel = intp(z)
        msk = ~np.isfinite(zsel)
        zsel[msk] = 0.0
        vals = np.zeros((m.shape[0], a.shape[0]))
        vals[:] = zsel
        msk = (logm >= 14) & (logm < 16)
        vals[~msk, :] = 0
        return vals

    mdef = ccl.halos.MassDef(200, "matter")
    hmf = ccl.halos.MassFuncTinker10(cosmo, mdef, mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef, mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(
        cosmo, hmf, hbf, mdef, integration_method_M="spline", nlog10M=256
    )

    true_cnts = hmc.number_counts(cosmo, _sel, amin=0.333333, amax=1, na=256)
    true_cnts *= 15.1

    sacc_data.add_data_point(
        "count",
        ("trc1",),
        true_cnts / 10,
    )

    assert true_cnts > 0

    def _src_sel(lnmass, a):
        return _sel(np.exp(lnmass), a)

    source = DummySource()
    source.sacc_tracer = "trc1"
    source.selfunc_ = _src_sel
    source.area_sr_ = 15.1
    source.z_ = z
    sources = {"trc11": source}

    stat = ClusterCountStatistic(
        ["trc11"],
        mass_def=[200, "matter"],
        mass_func="Tinker10",
        halo_bias="Tinker10",
        systematics=None,
        na=256,
        nlog10M=256,
    )
    stat.read(sacc_data, sources)
    stat.compute(cosmo, {}, sources)

    assert np.allclose(stat.predicted_statistic_, true_cnts)
    assert np.allclose(stat.measured_statistic_, true_cnts / 10)
