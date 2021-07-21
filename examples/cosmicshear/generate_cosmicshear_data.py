import numpy as np
import sacc
import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.045,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
    A_s=2.1e-9,
    n_s=0.96,
    h=0.67,
)

seed = 42
rng = np.random.RandomState(seed=seed)
eps = 0.01

sacc_data = sacc.Sacc()

tracers = []
for i, mn in enumerate([0.25, 0.75]):
    z = np.linspace(0, 2, 50) + 0.05
    dndz = np.exp(-0.5 * (z - mn) ** 2 / 0.25 / 0.25)

    sacc_data.add_tracer("NZ", "trc%d" % i, z, dndz)

    tracers.append(ccl.WeakLensingTracer(cosmo, dndz=(z, dndz)))

dv = []
ndv = []
for i in range(len(tracers)):
    for j in range(i, len(tracers)):
        ell = np.logspace(1, 4, 10)
        pell = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell)
        npell = pell + rng.normal(size=pell.shape[0]) * eps * pell

        sacc_data.add_ell_cl("galaxy_shear_cl_ee", "trc%d" % i, "trc%d" % j, ell, npell)
        dv.append(pell)
        ndv.append(npell)

# a fake covariance matrix
dv = np.concatenate(dv, axis=0)
cov = np.zeros((len(dv), len(dv)))
for i in range(len(dv)):
    cov[i, i] = (eps * dv[i]) ** 2

sacc_data.add_covariance(cov)

sacc_data.save_fits("cosmicshear.fits", overwrite=True)
