import os
import pandas as pd
import numpy as np
import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.045,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
    A_s=2.1e-9,
    n_s=0.96,
    h=0.67)

seed = 42
tmpdir = '.'
rng = np.random.RandomState(seed=seed)
eps = 0.01

tracers = []
for i, mn in enumerate([0.25, 0.75]):
    z = np.linspace(0, 2, 50) + 0.05
    dndz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)

    df = pd.DataFrame({'z': z, 'dndz': dndz})
    df.to_csv(
        os.path.join(tmpdir, 'pz%d.csv' % i),
        index=False)

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

        df = pd.DataFrame({'ell_or_theta': ell, 'measured_statistic': npell})
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
