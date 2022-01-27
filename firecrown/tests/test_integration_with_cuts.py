import os
import numpy as np
import pytest

import sacc
import pyccl as ccl

from ..parser import parse
from ..loglike import compute_loglike
from ..io import write_statistics


@pytest.fixture(scope="session")
def tx_data(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("data"))

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
        transfer_function='eisenstein_hu')

    seed = 42
    rng = np.random.RandomState(seed=seed)
    eps = 0.01

    tracers = []
    for i, mn in enumerate([0.25, 0.75]):
        z = np.linspace(0, 2, 50)
        dndz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)

        sacc_data.add_tracer('NZ', 'trc%d' % i, z, dndz)

        tracers.append(ccl.WeakLensingTracer(
            cosmo,
            dndz=(z, dndz)))

    ell_min = {}
    ell_max = {}

    dv = []
    ndv = []
    dv_orig = []
    ndv_orig = []
    inds = []
    msks = []
    for i in range(len(tracers)):
        for j in range(i, len(tracers)):
            ell = np.logspace(1, 4, 10)
            pell = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell)
            npell = pell + rng.normal(size=pell.shape[0]) * eps * pell

            # all of the data goes into the file
            sacc_data.add_ell_cl(
                'galaxy_shear_cl_ee', 'trc%d' % i, 'trc%d' % j, ell, npell)
            dv_orig.append(pell)
            ndv_orig.append(npell)
            inds.append(np.ones_like(pell))
            msk = np.ones_like(pell).astype(np.bool)

            # but only some of it comes back out
            if rng.uniform() < 0.5:
                ell_min[(i, j)] = rng.uniform(10, 100)
                msk &= (ell >= ell_min[(i, j)])
            else:
                ell_min[(i, j)] = None

            if rng.uniform() < 0.5:
                ell_max[(i, j)] = rng.uniform(1000, 10000)
                msk &= (ell <= ell_max[(i, j)])
            else:
                ell_max[(i, j)] = None

            if ell_min[(i, j)] is not None:
                q = np.where(ell >= ell_min[(i, j)])
                ell = ell[q]
                pell = pell[q]
                npell = npell[q]

            if ell_max[(i, j)] is not None:
                q = np.where(ell <= ell_max[(i, j)])
                ell = ell[q]
                pell = pell[q]
                npell = npell[q]

            msks.append(msk)

            dv.append(pell)
            ndv.append(npell)

    # a fake covariance matrix
    dv_orig = np.concatenate(dv_orig, axis=0)
    ndv_orig = np.concatenate(ndv_orig, axis=0)
    cov = np.zeros((dv_orig.shape[0], dv_orig.shape[0]))
    for i in range(len(dv_orig)):
        cov[i, i] = (eps * dv_orig[i]) ** 2
    sacc_data.add_covariance(cov)

    sacc_data.save_fits(os.path.join(tmpdir, 'sacc.fits'), overwrite=True)

    # cut the cov mat
    inds = np.concatenate(inds, axis=0)
    inds = np.cumsum(inds) - 1
    msks = np.where(np.concatenate(msks, axis=0))[0]
    n_keep = msks.shape[0]
    new_cov = np.zeros((n_keep, n_keep))
    for i_new, i_old in enumerate(msks):
        for j_new, j_old in enumerate(msks):
            new_cov[i_new, j_new] = cov[i_old, j_old]

    # compute loglike
    dv = np.concatenate(dv, axis=0)
    ndv = np.concatenate(ndv, axis=0)
    cinv = np.linalg.inv(new_cov)
    delta = ndv - dv
    loglike = -0.5 * np.dot(delta, np.dot(cinv, delta))

    config = """\
parameters:
  Omega_k: 0.0
  Omega_c: 0.27
  Omega_b: 0.045
  h: 0.67
  n_s: [0.9, 0.96, 1.0]
  sigma8: 0.8
  w0: -1.0
  wa: 0.0
  transfer_function: 'eisenstein_hu'

  # lens bin zero
  src0_delta_z: [-0.1, 0.0, 0.1]
  src1_delta_z: 0.0

two_point:
  module: firecrown.ccl.two_point
  sacc_data: {tmpdir}/sacc.fits
  sources:
    src0:
      kind: WLSource
      sacc_tracer: trc0
      systematics:
        - pz_delta_0

    src1:
      kind: WLSource
      sacc_tracer: trc1
      systematics:
        - pz_delta_1

  systematics:
    pz_delta_0:
      kind: PhotoZShiftBias
      delta_z: src0_delta_z

    pz_delta_1:
      kind: PhotoZShiftBias
      delta_z: src1_delta_z

  likelihood:
    kind: ConstGaussianLogLike
    data_vector:
      - cl_src0_src0
      - cl_src0_src1
      - cl_src1_src1

  statistics:
""".format(tmpdir=tmpdir)

    for i in range(len(tracers)):
        for j in range(i, len(tracers)):
            config += """\
    cl_src{i}_src{j}:
      sources: ['src{i}', 'src{j}']
      sacc_data_type: galaxy_shear_cl_ee
""".format(i=i, j=j)

            if ell_min[(i, j)] is not None:
                config += "      ell_or_theta_min: {val}\n".format(val=ell_min[(i, j)])

            if ell_max[(i, j)] is not None:
                config += "      ell_or_theta_max: {val}\n".format(val=ell_max[(i, j)])

    with open(os.path.join(tmpdir, 'config.yaml'), 'w') as fp:
        fp.write(config)

    return {
        'cosmo': cosmo,
        'tmpdir': tmpdir,
        'loglike': loglike,
        'config': config,
        'cov': new_cov,
        'inv_cov': cinv,
    }


def test_integration_with_cuts_smoke(tx_data):
    tmpdir = tx_data['tmpdir']
    cfg_path = os.path.join(tmpdir, 'config.yaml')

    config, data = parse(cfg_path)
    loglike, meas, pred, covs, inv_covs, stats = compute_loglike(
        cosmo=tx_data['cosmo'],
        data=data)

    assert np.allclose(loglike["two_point"], tx_data['loglike'])

    write_statistics(
        output_dir=os.path.join(tmpdir, 'output_123'),
        data=data,
        statistics=stats,
    )

    opth = os.path.join(tmpdir, 'output_123', 'statistics', 'two_point')
    orig_data = sacc.Sacc.load_fits(os.path.join(tmpdir, 'sacc.fits'))
    meas_data = sacc.Sacc.load_fits(os.path.join(opth, 'sacc_measured.fits'))
    pred_data = sacc.Sacc.load_fits(os.path.join(opth, 'sacc_predicted.fits'))

    for trc_name in ['trc0', 'trc1']:
        orig_tr = orig_data.get_tracer(trc_name)
        meas_tr = meas_data.get_tracer(trc_name)
        pred_tr = pred_data.get_tracer(trc_name)

        assert np.allclose(orig_tr.z, meas_tr.z)
        assert np.allclose(orig_tr.z, pred_tr.z)
        assert np.allclose(orig_tr.nz, meas_tr.nz)
        assert np.allclose(orig_tr.nz, pred_tr.nz)

    meas_dv = []
    pred_dv = []
    for trs in [('trc0', 'trc0'), ('trc0', 'trc1'), ('trc1', 'trc1')]:
        mell, mcl = meas_data.get_ell_cl('galaxy_shear_cl_ee', trs[0], trs[1])
        pell, pcl = pred_data.get_ell_cl('galaxy_shear_cl_ee', trs[0], trs[1])

        assert np.allclose(pell, mell)
        assert not np.array_equal(mcl, pcl)
        meas_dv.append(mcl)
        pred_dv.append(pcl)

    assert np.allclose(np.concatenate(meas_dv, axis=0), meas["two_point"])
    assert np.allclose(np.concatenate(pred_dv, axis=0), pred["two_point"])
    assert np.allclose(covs["two_point"], tx_data["cov"])
    assert np.allclose(inv_covs["two_point"], tx_data["inv_cov"])
