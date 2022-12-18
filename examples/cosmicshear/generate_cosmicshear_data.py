import numpy as np
import sacc
import pyccl as ccl

COSMO = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.045,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
    A_s=2.1e-9,
    n_s=0.96,
    h=0.67,
)

SEED = 42
EPS = 0.01

Z = np.linspace(0, 2, 50) + 0.05
ELL = np.logspace(1, 4, 10)

rng = np.random.RandomState(seed=SEED)
sacc_data = sacc.Sacc()
tracers = []

for i, mn in enumerate([0.25, 0.75]):
    dndz = np.exp(-0.5 * (Z - mn) ** 2 / 0.25 / 0.25)
    sacc_data.add_tracer("NZ", f"trc{i}", Z, dndz)
    tracers.append(ccl.WeakLensingTracer(COSMO, dndz=(Z, dndz)))

dv = []

# Fill in the upper triangular indices for dv.
for i, j in zip(*np.triu_indices(len(tracers))):
    pell = ccl.angular_cl(COSMO, tracers[i], tracers[j], ELL)
    npell = pell + rng.normal(size=pell.shape[0]) * EPS * pell
    sacc_data.add_ell_cl("galaxy_shear_cl_ee", f"trc{i}", f"trc{j}", ELL, npell)
    dv.append(pell)


# a fake covariance matrix
dv = np.concatenate(dv, axis=0)
cov = np.diag((EPS * dv) ** 2)

sacc_data.add_covariance(cov)
sacc_data.save_fits("cosmicshear.fits", overwrite=True)
