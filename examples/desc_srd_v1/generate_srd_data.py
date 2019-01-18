import os

import numpy as np
import pandas as pd

from srd_models import (
    make_src_z_bins, make_lens_z_bins,
    make_src_ell_bins, make_lens_ell_bins,
    make_lens_src_ell_bins,
    N_BINS)


os.makedirs('data', exist_ok=True)

make_src_z_bins('data')
make_lens_z_bins('data')
make_src_ell_bins('data')

lens_mean_z = []
for i in range(N_BINS):
    df = pd.read_csv(os.path.join('data', 'lens%d_dndz.csv' % i))
    lens_mean_z.append(np.sum(df['z'] * df['dndz']) / np.sum(df['dndz']))
lens_mean_z = np.array(lens_mean_z)

make_lens_ell_bins('data', lens_mean_z)
make_lens_src_ell_bins('data', lens_mean_z)
