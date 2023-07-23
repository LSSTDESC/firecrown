import os
import pyccl as ccl
from typing import Any, Dict
import numpy as np

from firecrown.models.cluster_abundance import ClusterAbundance
from firecrown.models.cluster_mass_rich_proxy import (
    ClusterMassRich,
    ClusterMassRichBinArgument,
    ClusterMassRichPointArgument,
)
from firecrown.models.cluster_redshift_spec import (
    ClusterRedshiftSpec,
    ClusterRedshiftSpecArgument,
)
from firecrown.models.cluster_mass import ClusterMassArgument
from firecrown.models.cluster_redshift import ClusterRedshiftArgument
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import time
from numcosmo_py import Ncm

Omega_c = 0.262
Omega_b = 0.049
Omega_k = 0.0
H0 = 67.66
h = H0 / 100.0
Tcmb0 = 2.7255
A_s = 2.1e-9
sigma8 = 0.8277
n_s = 0.96
Neff = 3.046
w0 = -1.0
wa = 0.0
cosmo_ccl = ccl.Cosmology(
    Omega_c=Omega_c,
    Omega_b=Omega_b,
    Neff=Neff,
    h=h,
    sigma8=sigma8,
    n_s=n_s,
    Omega_k=Omega_k,
    w0=w0,
    wa=wa,
    T_CMB=Tcmb0,
    m_nu=[0.00, 0.0, 0.0],
)
pivot_mass = 14.625862906
pivot_redshift = 0.6

cluster_mass_r = ClusterMassRich(pivot_mass, pivot_redshift)

cluster_mass_r.__setattr__("mu_p0", 3.0)
cluster_mass_r.__setattr__("mu_p1", 0.86)
cluster_mass_r.__setattr__("mu_p2", 0.0)
cluster_mass_r.__setattr__("sigma_p0", 3.0)
cluster_mass_r.__setattr__("sigma_p1", 0.7)
cluster_mass_r.__setattr__("sigma_p2", 0.0)

cluster_z = ClusterRedshiftSpec()

try:
    hmd_200 = ccl.halos.MassDef200m()
except TypeError:
    hmd_200 = ccl.halos.MassDef200m

hmf_args: Dict[str, Any] = {}
hmf_name = "Tinker08"
z_bins = np.linspace(0.0, 1.0, 4)
r_bins = np.linspace(1.0, 2.5, 5)


integ_options = [
    Ncm.IntegralNDMethod.P,
    Ncm.IntegralNDMethod.P_V,
    Ncm.IntegralNDMethod.H,
    Ncm.IntegralNDMethod.H_V,
]
for integ_method in integ_options:
    abundance_test = ClusterAbundance(
        hmd_200, hmf_name, hmf_args, integ_method=integ_method
    )
    test_m_list = []
    t1 = time.time()
    for i in range(0, 3):
        for j in range(0, 4):
            bin_ij = ((z_bins[i], z_bins[i + 1]), (r_bins[j], r_bins[j + 1]))
            cluster_mass_bin = ClusterMassRichBinArgument(
                cluster_mass_r, 13, 15, r_bins[j], r_bins[j + 1]
            )
            cluster_z_bin = ClusterRedshiftSpecArgument(z_bins[i], z_bins[i + 1])
            cluster_counts = abundance_test.compute(
                cosmo_ccl, cluster_mass_bin, cluster_z_bin
            )
            test_m_list.append(cluster_counts)
    t2 = time.time()

    print(
        f"The time for {integ_method} is {t2-t1}\n\n The counts value is {test_m_list}\n\n"
    )
