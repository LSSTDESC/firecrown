import os

import numpy as np
import pandas as pd

from srd_models import (
    make_src_z_bins, make_lens_z_bins,
    make_src_ell_bins, make_lens_ell_bins,
    make_lens_src_ell_bins,
    N_BINS)


os.makedirs('data_gen', exist_ok=True)
os.makedirs('data_model', exist_ok=True)

for dr in ['data_gen', 'data_model']:
    make_src_z_bins(dr)
    make_lens_z_bins(dr)

make_src_ell_bins('data_gen')

lens_mean_z = []
for i in range(N_BINS):
    df = pd.read_csv(os.path.join('data_gen', 'lens%d_dndz.csv' % i))
    lens_mean_z.append(np.sum(df['z'] * df['dndz']) / np.sum(df['dndz']))
lens_mean_z = np.array(lens_mean_z)

make_lens_ell_bins('data_gen', lens_mean_z)
make_lens_src_ell_bins('data_gen', lens_mean_z)

# code to make the cov mat
cov_file = (
    'LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/cov'
    '/Y1_3x2pt_clusterN_clusterWL_cov')
data_file = (
    'LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/datav'
    '/3x2pt_clusterN_clusterWL_Y1_fid')

if os.path.exists(cov_file):
    nggl = 7  # number of ggl power spectra
    ngcl = 6  # number of cluster-source galaxy power spectra
    nlens = 5  # number of lens bins
    nlenscl = 3  # number of cluster redshift bins
    nshear = 15  # number of shear tomographic power spectra
    ncl = 20  # number of ell-bins
    nclgcl = 5  # number of cluster ell-bins
    nrich = 5  # number of richness bins

    ndata = (
        (nshear + nggl + nlens) * ncl +
        nlenscl * nrich +
        nrich * ngcl * nclgcl)
    n2pt = (nshear + nggl + nlens) * ncl

    datav = np.genfromtxt(data_file)
    mask = np.zeros(ndata)
    for i in range(datav.shape[0]):
        if datav[i, 1] > 1.0e-15:
            mask[i] = 1.0
    mask = mask.astype(bool)

    covfile = np.genfromtxt(cov_file)
    cov = np.zeros((ndata, ndata))

    for i in range(covfile.shape[0]):
        iind, jind = int(covfile[i, 0]), int(covfile[i, 1])
        val = covfile[i, 8] + covfile[i, 9]
        cov[iind, jind] = val
        cov[jind, iind] = val

    mask = mask[:n2pt]
    cumsum_mask = np.cumsum(mask) - 1
    cov = cov[:n2pt, :n2pt]
    datav = datav[:n2pt]

    cor = np.zeros_like(cov)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if cov[i, i] * cov[j, j] > 0:
                cor[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

    with open('srd_data/Y1_3x2pt_clusterN_clusterWL_cov.csv', 'w') as fp:
        fp.write('i,j,cov\n')
        for _i in range(cov.shape[0]):
            for _j in range(cov.shape[1]):
                if mask[_i] and mask[_j]:
                    i = cumsum_mask[_i]
                    j = cumsum_mask[_j]
                    fp.write('%d,%d,%.16e\n' % (i, j, cov[_i, _j]))

    with open('srd_data/Y1_3x2pt_clusterN_clusterWL_fid.csv', 'w') as fp:
        fp.write('i,val\n')
        for _i in range(datav.shape[0]):
            fp.write('%d,%.16e\n' % (_i, datav[_i, 1]))
