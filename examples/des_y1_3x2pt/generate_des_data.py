import os
import fitsio
import pandas as pd
import numpy as np


os.makedirs('des_data', exist_ok=True)

# bin limits are from the chain headers courtesy of M. Troxel
angles = """\
## angle_range_xip_1_1 = 7.195005 250.0
## angle_range_xip_1_2 = 7.195005 250.0
## angle_range_xip_1_3 = 5.715196 250.0
## angle_range_xip_1_4 = 5.715196 250.0
## angle_range_xip_2_1 = 7.195005 250.0
## angle_range_xip_2_2 = 4.539741 250.0
## angle_range_xip_2_3 = 4.539741 250.0
## angle_range_xip_2_4 = 4.539741 250.0
## angle_range_xip_3_1 = 5.715196 250.0
## angle_range_xip_3_2 = 4.539741 250.0
## angle_range_xip_3_3 = 3.606045 250.0
## angle_range_xip_3_4 = 3.606045 250.0
## angle_range_xip_4_1 = 5.715196 250.0
## angle_range_xip_4_2 = 4.539741 250.0
## angle_range_xip_4_3 = 3.606045 250.0
## angle_range_xip_4_4 = 3.606045 250.0
## angle_range_xim_1_1 = 90.579750 250.0
## angle_range_xim_1_2 = 71.950053 250.0
## angle_range_xim_1_3 = 71.950053 250.0
## angle_range_xim_1_4 = 71.950053 250.0
## angle_range_xim_2_1 = 71.950053 250.0
## angle_range_xim_2_2 = 57.151958 250.0
## angle_range_xim_2_3 = 57.151958 250.0
## angle_range_xim_2_4 = 45.397414 250.0
## angle_range_xim_3_1 = 71.950053 250.0
## angle_range_xim_3_2 = 57.151958 250.0
## angle_range_xim_3_3 = 45.397414 250.0
## angle_range_xim_3_4 = 45.397414 250.0
## angle_range_xim_4_1 = 71.950053 250.0
## angle_range_xim_4_2 = 45.397414 250.0
## angle_range_xim_4_3 = 45.397414 250.0
## angle_range_xim_4_4 = 36.060448 250.0
## angle_range_gammat_1_1 = 64.0 250.0
## angle_range_gammat_1_2 = 64.0 250.0
## angle_range_gammat_1_3 = 64.0 250.0
## angle_range_gammat_1_4 = 64.0 250.0
## angle_range_gammat_2_1 = 40.0 250.0
## angle_range_gammat_2_2 = 40.0 250.0
## angle_range_gammat_2_3 = 40.0 250.0
## angle_range_gammat_2_4 = 40.0 250.0
## angle_range_gammat_3_1 = 30.0 250.0
## angle_range_gammat_3_2 = 30.0 250.0
## angle_range_gammat_3_3 = 30.0 250.0
## angle_range_gammat_3_4 = 30.0 250.0
## angle_range_gammat_4_1 = 24.0 250.0
## angle_range_gammat_4_2 = 24.0 250.0
## angle_range_gammat_4_3 = 24.0 250.0
## angle_range_gammat_4_4 = 24.0 250.0
## angle_range_gammat_5_1 = 21.0 250.0
## angle_range_gammat_5_2 = 21.0 250.0
## angle_range_gammat_5_3 = 21.0 250.0
## angle_range_gammat_5_4 = 21.0 250.0
## angle_range_wtheta_1_1 = 43.0 250.0
## angle_range_wtheta_2_2 = 27.0 250.0
## angle_range_wtheta_3_3 = 20.0 250.0
## angle_range_wtheta_4_4 = 16.0 250.0
## angle_range_wtheta_5_5 = 14.0 250.0"""

# here we munge them to a dict of dicts with structure:
#
# {'xip': {(1, 1): [7.195005, 250.0],
#   (1, 2): [7.195005, 250.0],
#   (1, 3): [5.715196, 250.0],
#   ...
#  'xim': {(1, 1): [90.57975, 250.0],
#   (1, 2): [71.950053, 250.0],
#   (1, 3): [71.950053, 250.0],
#   ...
#  'gammat': {(1, 1): [64.0, 250.0],
#   (1, 2): [64.0, 250.0],
#   (1, 3): [64.0, 250.0],
#   ...
#  'wtheta': {(1, 1): [43.0, 250.0],
#   (2, 2): [27.0, 250.0],
#   (3, 3): [20.0, 250.0],
#   ...

bin_limits = {}
for line in angles.split('\n'):
    items = line.split()
    keys = items[1].replace('angle_range_', '').split('_')
    topkey = keys[0]
    binkeys = (int(keys[1]), int(keys[2]))
    if topkey not in bin_limits:
        bin_limits[topkey] = {}
    bin_limits[topkey][binkeys] = (float(items[-2]), float(items[-1]))

# finally we read the data, cut each part, and write to disk
# the order of the covmat is xip, xim, gammat, wtheta
# these elements range from
#   xip: [0, 200)
#   xip: [200, 400)
#   gammat: [400, 800)
#   wtheta: [800, 900)
# there are 20 angular bins per data vector
# there are 4 source bins
# there are 5 lens bins
# only the autocorrelation wtheta bins are kept
n_srcs = 4
n_lens = 5

# this holds a global mask of which elements of the data vector to keep
tot_msk = []

with fitsio.FITS('2pt_NG_mcal_1110.fits') as data:
    # xip
    xip = data['xip'].read()
    for i in range(1, n_srcs+1):
        for j in range(i, n_srcs+1):
            theta_min, theta_max = bin_limits['xip'][(i, j)]

            ij_msk = (xip['BIN1'] == i) & (xip['BIN2'] == j)
            xip_ij = xip[ij_msk]
            msk = (xip_ij['ANG'] > theta_min) & (xip_ij['ANG'] < theta_max)

            tot_msk.extend(msk.tolist())

            df = pd.DataFrame({
                'ell_or_theta': xip_ij['ANG'][msk],
                'measured_statistic': xip_ij['VALUE'][msk]})
            df.to_csv(
                'des_data/xip_src%d_src%d.csv' % (i, j),
                index=False)

    # xim
    xim = data['xim'].read()
    for i in range(1, n_srcs+1):
        for j in range(i, n_srcs+1):
            theta_min, theta_max = bin_limits['xim'][(i, j)]

            ij_msk = (xim['BIN1'] == i) & (xim['BIN2'] == j)
            xim_ij = xim[ij_msk]
            msk = (xim_ij['ANG'] > theta_min) & (xim_ij['ANG'] < theta_max)

            tot_msk.extend(msk.tolist())

            df = pd.DataFrame({
                'ell_or_theta': xim_ij['ANG'][msk],
                'measured_statistic': xim_ij['VALUE'][msk]})
            df.to_csv(
                'des_data/xim_src%d_src%d.csv' % (i, j),
                index=False)

    # gammat
    gammat = data['gammat'].read()
    for i in range(1, n_lens+1):
        for j in range(1, n_srcs+1):
            theta_min, theta_max = bin_limits['gammat'][(i, j)]

            ij_msk = (gammat['BIN1'] == i) & (gammat['BIN2'] == j)
            gammat_ij = gammat[ij_msk]
            msk = (
                (gammat_ij['ANG'] > theta_min) &
                (gammat_ij['ANG'] < theta_max))

            tot_msk.extend(msk.tolist())

            df = pd.DataFrame({
                'ell_or_theta': gammat_ij['ANG'][msk],
                'measured_statistic': gammat_ij['VALUE'][msk]})
            df.to_csv(
                'des_data/gammat_lens%d_src%d.csv' % (i, j),
                index=False)

    # wtheta
    wtheta = data['wtheta'].read()
    for i in range(1, n_lens+1):
        theta_min, theta_max = bin_limits['wtheta'][(i, i)]

        ii_msk = (wtheta['BIN1'] == i) & (wtheta['BIN2'] == i)
        wtheta_ii = wtheta[ii_msk]
        msk = (
            (wtheta_ii['ANG'] > theta_min) &
            (wtheta_ii['ANG'] < theta_max))

        tot_msk.extend(msk.tolist())

        df = pd.DataFrame({
            'ell_or_theta': wtheta_ii['ANG'][msk],
            'measured_statistic': wtheta_ii['VALUE'][msk]})
        df.to_csv(
            'des_data/wtheta_lens%d_lens%d.csv' % (i, i),
            index=False)

    # nz_lens
    dndz = data['nz_lens'].read()
    for i in range(1, n_lens+1):
        df = pd.DataFrame({
            'z': dndz['Z_MID'],
            'dndz': dndz['BIN%d' % i]})
        df.to_csv(
            'des_data/dndz_lens%d.csv' % i,
            index=False)

    # nz_src
    dndz = data['nz_source'].read()
    for i in range(1, n_srcs+1):
        df = pd.DataFrame({
            'z': dndz['Z_MID'],
            'dndz': dndz['BIN%d' % i]})
        df.to_csv(
            'des_data/dndz_src%d.csv' % i,
            index=False)

    # covmat
    msk_inds = np.where(tot_msk)[0]
    n_cov = np.sum(tot_msk)
    old_cov = data['COVMAT'].read()
    cov_i = []
    cov_j = []
    new_cov = []
    for new_cov_i, old_cov_i in enumerate(msk_inds):
        for new_cov_j, old_cov_j in enumerate(msk_inds):
            new_cov.append(old_cov[old_cov_i, old_cov_j])
            cov_i.append(new_cov_i)
            cov_j.append(new_cov_j)
    df = pd.DataFrame({
        'i': cov_i,
        'j': cov_j,
        'cov': new_cov})
    df.to_csv(
        'des_data/cov.csv',
        index=False)

# now tar the data
os.system('tar czvf des_data.tar.gz des_data')
os.system('rm -rf des_data')
