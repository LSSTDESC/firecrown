import os
import tempfile

import pandas as pd
import numpy as np
import scipy.linalg

from ..tdist import TdistLogLike


def test_gaussian():
    n = 5
    rng = np.random.RandomState(seed=42)
    # code to make a random positive semi-definite, symmetric matrix
    cov = rng.rand(n, n)
    u, s, v = scipy.linalg.svd(np.dot(cov.T, cov))
    cov = np.dot(np.dot(u, 1.0 + np.diag(rng.rand(n))), v)
    delta = rng.normal(size=n)
    nu = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        rows = []
        for i in range(n):
            for j in range(n):
                rows.append((i, j, cov[i, j]))
        df = pd.DataFrame.from_records(rows, columns=['i', 'j', 'cov'])

        fname = os.path.join(tmpdir, 'cov.dat')
        df.to_csv(fname, index=False)

        ll = TdistLogLike(
            data=fname,
            data_vector=['a'],
            nu=nu)

    assert ll.data_vector == ['a']
    assert ll.nu == nu
    chi2 = np.dot(delta, np.dot(np.linalg.inv(cov), delta))
    loglike = -0.5 * nu * np.log(1.0 + chi2 / (nu - 1.0))
    data = {'a': delta}
    theory = {'a': np.zeros(n)}
    assert np.allclose(loglike, ll.compute(data, theory))
