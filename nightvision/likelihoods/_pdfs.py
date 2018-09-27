import numpy as np
import pandas as pd
import scipy.linalg


def parse_gauss_pdf(keys):
    new_keys = {}
    df = pd.read_csv(keys['data'])
    dim = max(np.max(df['i']), np.max(df['j'])) + 1
    cov = np.zeros((dim, dim))
    cov[df['i'].values, df['j'].values] = df['cov'].values
    new_keys['cov'] = cov
    new_keys['L'] = np.linalg.cholesky(cov)
    new_keys.update(keys)

    def _comp_ll(dv):
        x = scipy.linalg.solve_triangular(new_keys['L'], dv)
        loglike = -0.5 * np.dot(x, x)
        return loglike

    new_keys['comp'] = _comp_ll

    return new_keys
