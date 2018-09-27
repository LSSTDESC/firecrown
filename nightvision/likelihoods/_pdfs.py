import numpy as np
import pandas as pd
import scipy.linalg


def parse_gaussian_pdf(keys):
    new_keys = {}
    df = pd.read_csv(keys['data'])
    dim = max(np.max(df['i']), np.max(df['j'])) + 1
    cov = np.zeros((dim, dim))
    cov[df['i'].values, df['j'].values] = df['cov'].values
    new_keys['cov'] = cov
    new_keys['L'] = np.linalg.cholesky(cov)
    new_keys.update(keys)
    return new_keys


def compute_gaussian_pdf(dv, L):
    x = scipy.linalg.solve_triangular(L, dv)
    return -0.5 * np.dot(x, x)
