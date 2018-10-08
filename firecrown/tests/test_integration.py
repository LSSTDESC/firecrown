import os

import pandas as pd
import numpy as np

import pytest

import pyccl as ccl

from ..parser import parse
from ..loglike import compute_loglike


@pytest.fixture(scope="session")
def tx_data(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("data"))

    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        sigma8=0.8,
        n_s=0.96,
        h=0.67)

    seed = 42
    rng = np.random.RandomState(seed=seed)
    eps = 0.01

    tracers = []
    for i, mn in enumerate([0.25, 0.75]):
        z = np.linspace(0, 2, 50)
        nz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)

        df = pd.DataFrame({'z': z, 'nz': nz})
        df.to_csv(
            os.path.join(tmpdir, 'pz%d.csv' % i),
            index=False)

        tracers.append(ccl.ClTracerLensing(
            cosmo,
            has_intrinsic_alignment=False,
            z=z,
            n=nz))

    dv = []
    ndv = []
    for i in range(len(tracers)):
        for j in range(i, len(tracers)):
            ell = np.logspace(1, 4, 10)
            pell = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell)
            npell = pell + rng.normal(size=pell.shape[0]) * eps * pell

            df = pd.DataFrame({'l': ell, 'cl': npell})
            df.to_csv(
                os.path.join(tmpdir, 'cl%d%d.csv' % (i, j)),
                index=False)
            dv.append(pell)
            ndv.append(npell)

    # a fake covariance matrix
    dv = np.concatenate(dv, axis=0)
    ndv = np.concatenate(ndv, axis=0)
    nelts = len(tracers) * (len(tracers) + 1) // 2
    cov = np.identity(len(pell) * nelts)
    for i in range(len(dv)):
        cov[i, i] *= (eps * dv[i]) ** 2
    assert len(dv) == cov.shape[0]
    _i = []
    _j = []
    _val = []
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            _i.append(i)
            _j.append(j)
            _val.append(cov[i, j])
    df = pd.DataFrame({'i': _i, 'j': _j, 'cov': _val})
    df.to_csv(
        os.path.join(tmpdir, 'cov.csv'),
        index=False)

    cinv = np.linalg.inv(cov)
    delta = ndv - dv
    loglike = -0.5 * np.dot(delta, np.dot(cinv, delta))

    config = """\
parameters:
  Omega_k: 0.0
  Omega_c: 0.27
  Omega_b: 0.045
  h: 0.67
  n_s: 0.96
  sigma_8: 0.8
  w0: -1.0
  wa: 0.0

  # lens bin zero
  src0_delta_z: 0.0
  src1_delta_z: 0.0

two_point:
  module: firecrown.ccl.two_point
  sources:
    src0:
      kind: ClTracerLensing
      nz_data: {tmpdir}/pz0.csv
      has_intrinsic_alignment: False

      systematics:
        photoz_shift:
          delta_z: src0_delta_z

    src1:
      kind: ClTracerLensing
      nz_data: {tmpdir}/pz1.csv
      has_intrinsic_alignment: False

      systematics:
        photoz_shift:
          delta_z: src1_delta_z

  likelihood:
    kind: gaussian
    data: {tmpdir}/cov.csv
    data_vector:
      - cl_src0_src0
      - cl_src0_src1
      - cl_src1_src1

  statistics:
    cl_src0_src0:
      sources: ['src0', 'src0']
      kind: 'cl'
      data: {tmpdir}/cl00.csv

    cl_src0_src1:
      sources: ['src0', 'src1']
      kind: 'cl'
      data: {tmpdir}/cl01.csv

    cl_src1_src1:
      sources: ['src1', 'src1']
      kind: 'cl'
      data: {tmpdir}/cl11.csv""".format(tmpdir=tmpdir)

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
