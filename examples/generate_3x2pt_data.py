import os
import numpy as np
import pandas as pd
import pyccl as ccl

seed = 42
rng = np.random.RandomState(seed=seed)
eps = 0.01

params = ccl.Parameters(
    Omega_c=0.27,
    Omega_b=0.045,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
    sigma8=0.8,
    n_s=0.96,
    h=0.67)
cosmo = ccl.Cosmology(params)

os.makedirs('data', exist_ok=True)

tracers = []
for i, mn in enumerate([0.25, 0.75]):
    z = np.linspace(0, 2, 50)
    nz = np.exp(-0.5 * (z - mn)**2 / 0.25 / 0.25)

    df = pd.DataFrame({'z': z, 'nz': nz})
    df.to_csv('./data/pz%d.csv' % i, index=False)

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
        df.to_csv('./data/cl%d%d.csv' % (i, j), index=False)
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
df.to_csv('./data/cov.csv', index=False)

cinv = np.linalg.inv(cov)
delta = ndv - dv
loglike = -0.5 * np.dot(delta, np.dot(cinv, delta))

print('chi2:', loglike)
