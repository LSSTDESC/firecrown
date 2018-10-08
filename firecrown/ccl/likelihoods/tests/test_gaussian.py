import os
import tempfile

import pandas as pd
import numpy as np
import scipy.linalg

from ..gaussian import ConstGaussianLogLike


def test_gaussian():
    n = 5
    rng = np.random.RandomState(seed=42)
    # code to make a random positive semi-definite, symmetric matrix
    cov = rng.rand(n, n)
    u, s, v = scipy.linalg.svd(np.dot(cov.T, cov))
    cov = np.dot(np.dot(u, 1.0 + np.diag(rng.rand(n))), v)
    delta = rng.normal(size=n)

    with tempfile.TemporaryDirectory() as tmpdir:
        rows = []
        for i in range(n):
            for j in range(n):
                rows.append((i, j, cov[i, j]))
        df = pd.DataFrame.from_records(rows, columns=['i', 'j', 'cov'])

        fname = os.path.join(tmpdir, 'cov.dat')
        df.to_csv(fname, index=False)

        ll = ConstGaussianLogLike(
            data=fname,
            data_vector=['a'])

    assert ll.data_vector == ['a']
    data = {'a': delta}
    theory = {'a': np.zeros(n)}
    loglike = -0.5 * np.dot(delta, np.dot(np.linalg.inv(cov), delta))
    assert np.allclose(loglike, ll.compute_loglike(data, theory))
