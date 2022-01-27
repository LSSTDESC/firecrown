import numpy as np

import sacc

from srd_models import (
    add_srci_lensj_ell_cl,
    add_lensi_lensi_ell_cl,
    add_srci_srcj_ell_cl,
    add_lens_tracers,
    add_src_tracers
)

sacc_data = sacc.Sacc()

####################################
# first lets add the tracers
add_src_tracers(sacc_data)
add_lens_tracers(sacc_data)

####################################
# now we add the data points in
# the correct order and the cov mat

cov_file = (
    'LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/cov'
    '/Y1_3x2pt_clusterN_clusterWL_cov')
data_file = (
    'LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/datav'
    '/3x2pt_clusterN_clusterWL_Y1_fid')

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

covfile = np.genfromtxt(cov_file)
cov = np.zeros((ndata, ndata))
for i in range(covfile.shape[0]):
    iind, jind = int(covfile[i, 0]), int(covfile[i, 1])
    val = covfile[i, 8] + covfile[i, 9]
    cov[iind, jind] = val
    cov[jind, iind] = val

cov = cov[:n2pt, :n2pt]
datav = datav[:n2pt]

msks = []

# shear-shear
loc = 0
for i in range(5):
    for j in range(i, 5):
        msk = add_srci_srcj_ell_cl(sacc_data, i, j, datav[loc:loc+ncl, 1])
        msks.append(msk)
        loc += ncl

# shear-gal
# i is the source index
# j is the lens index
for j, i in [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4), (3, 4)]:
    tr = sacc_data.get_tracer('lens%d' % j)
    mean_z = np.sum(tr.z * tr.nz) / np.sum(tr.nz)
    msk = add_srci_lensj_ell_cl(sacc_data, i, j, mean_z, datav[loc:loc+ncl, 1])
    msks.append(msk)
    loc += ncl

# gal-gal
for i in range(5):
    tr = sacc_data.get_tracer('lens%d' % j)
    mean_z = np.sum(tr.z * tr.nz) / np.sum(tr.nz)
    msk = add_lensi_lensi_ell_cl(sacc_data, i, mean_z, datav[loc:loc+ncl, 1])
    msks.append(msk)
    loc += ncl

# make sure the masks are the right size
mask = np.concatenate(msks)
assert mask.shape[0] == datav.shape[0]

# make sure we got everything
assert loc == datav.shape[0]

# deal with the covariances
mask_inds = np.cumsum(mask) - 1
n_keep = np.sum(mask)
masked_cov = np.zeros((n_keep, n_keep))
for _i in range(cov.shape[0]):
    for _j in range(cov.shape[1]):
        if mask[_i] and mask[_j]:
            i = mask_inds[_i]
            j = mask_inds[_j]
            masked_cov[i, j] = cov[_i, _j]

sacc_data.add_covariance(masked_cov)

sacc_data.save_fits('srd_v1_sacc_data.fits', overwrite=True)
