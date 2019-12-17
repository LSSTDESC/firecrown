import os
import numpy as np
import pytest

import sacc
import pyccl as ccl

from ..parser import parse
from ..loglike import compute_loglike


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

    dv = []
    ndv = []
    for i in range(len(tracers)):
        for j in range(i, len(tracers)):
            ell = np.logspace(1, 4, 10)
            pell = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell)
            npell = pell + rng.normal(size=pell.shape[0]) * eps * pell

            sacc_data.add_ell_cl(
                'galaxy_shear_cl_ee', 'trc%d' % i, 'trc%d' % j, ell, npell)
            dv.append(pell)
            ndv.append(npell)

    # a fake covariance matrix
    dv = np.concatenate(dv, axis=0)
    ndv = np.concatenate(ndv, axis=0)
    cov = np.zeros((dv.shape[0], dv.shape[0]))
    for i in range(len(dv)):
        cov[i, i] = (eps * dv[i]) ** 2
    sacc_data.add_covariance(cov)

    sacc_data.save_fits(os.path.join(tmpdir, 'sacc.fits'), overwrite=True)

    cinv = np.linalg.inv(cov)
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
  sacc_file: {tmpdir}/sacc.fits
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
    cl_src0_src0:
      sources: ['src0', 'src0']
      sacc_data_type: galaxy_shear_cl_ee

    cl_src0_src1:
      sources: ['src0', 'src1']
      sacc_data_type: galaxy_shear_cl_ee

    cl_src1_src1:
      sources: ['src1', 'src1']
      sacc_data_type: galaxy_shear_cl_ee

""".format(tmpdir=tmpdir)

    with open(os.path.join(tmpdir, 'config.yaml'), 'w') as fp:
        fp.write(config)

    return {
        'cosmo': cosmo,
        'tmpdir': tmpdir,
        'loglike': loglike,
        'config': config}


def test_integration_smoke(tx_data):
    tmpdir = tx_data['tmpdir']
    cfg_path = os.path.join(tmpdir, 'config.yaml')

    config, data = parse(cfg_path)
    loglike, _ = compute_loglike(
        cosmo=tx_data['cosmo'],
        data=data)

    assert np.allclose(loglike, tx_data['loglike'])
