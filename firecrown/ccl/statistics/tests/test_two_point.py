import numpy as np
import pandas as pd
import os
import pytest
import pyccl as ccl

from ..two_point import TwoPointStatistic, _ell_for_xi


class DummySource(object):
    pass


@pytest.mark.parametrize(
    'kind',
    ['cl',  'gg', 'gl', 'l+', 'l-'])
def test_two_point_some(kind, tmpdir):

    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67)

    sources = {}
    for i, mn in enumerate([0.25, 0.75]):
        sources['src%d' % i] = DummySource()
        z = np.linspace(0, 2, 50)
        dndz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)

        if ('g' in kind and i == 0) or kind == 'gg':
            sources['src%d' % i].tracer_ = ccl.NumberCountsTracer(
                cosmo,
                has_rsd=False,
                dndz=(z, dndz),
                bias=(z, np.ones_like(z) * 2.0))
        else:
            sources['src%d' % i].tracer_ = ccl.WeakLensingTracer(
                cosmo,
                dndz=(z, dndz))

        sources['src%d' % i].scale_ = i / 2.0 + 1.0

    # compute the statistic
    tracers = [v.tracer_ for k, v in sources.items()]
    scale = np.prod([v.scale_ for k, v in sources.items()])
    data = os.path.join(tmpdir, 'stat.csv')
    if kind == 'cl':
        ell = np.logspace(1, 3, 10)
        cell = ccl.angular_cl(cosmo, *tracers, ell) * scale
        pd.DataFrame({'ell_or_theta': ell, 'measured_statistic': cell}).to_csv(
            data, index=False)
    else:
        theta = np.logspace(1, 2, 100)
        ell = _ell_for_xi()
        cell = ccl.angular_cl(cosmo, *tracers, ell)
        xi = ccl.correlation(
            cosmo, ell, cell, theta / 60.0, corr_type=kind) * scale
        pd.DataFrame({'ell_or_theta': theta, 'measured_statistic': xi}).to_csv(
            data, index=False)

    stat = TwoPointStatistic(
        data=data, kind=kind, sources=['src0', 'src1'])
    stat.compute(cosmo, {}, sources, systematics=None)

    if kind == 'cl':
        assert np.allclose(stat.measured_statistic_, cell)
    else:
        assert np.allclose(stat.measured_statistic_, xi)

    assert np.allclose(stat.measured_statistic_, stat.predicted_statistic_)
