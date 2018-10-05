# import os
# import tempfile
#
# import pandas as pd
# import numpy as np
# import scipy.linalg
#
# from ..gaussian import GaussianLikelihood
#
#
# def test_gaussian():
#     n = 5
#     rng = np.random.RandomState(seed=42)
#     # code to make a random positive semi-definite, symmetric matrix
#     cov = rng.rand(n, n)
#     u, s, v = scipy.linalg.svd(np.dot(cov.T, cov))
#     cov = np.dot(np.dot(u, 1.0 + np.diag(rng.rand(n))), v)
#     delta = rng.normal(size=n)
#
#     with tempfile.TemporaryDirectory() as tmpdir:
#         rows = []
#         for i in range(n):
#             for j in range(n):
#                 rows.append((i, j, cov[i, j]))
#         df = pd.DataFrame.from_records(rows, columns=['i', 'j', 'cov'])
#
#         fname = os.path.join(tmpdir, 'cov.dat')
#         df.to_csv(fname, index=False)
#
#         lk = GaussianLikelihood(covariance=fname, data_vector=['a', 'b'])
#
#     assert lk.data_vector == ['a', 'b']
#     loglike = -0.5 * np.dot(delta, np.dot(np.linalg.inv(cov), delta))
#     assert np.allclose(loglike, lk(delta, np.zeros(n)))
#
#
# import os
# import tempfile
#
# import pandas as pd
# import numpy as np
# import scipy.linalg
#
# from ..tdist import TDistLikelihood
#
#
# def test_gaussian():
#     n = 5
#     rng = np.random.RandomState(seed=42)
#     # code to make a random positive semi-definite, symmetric matrix
#     cov = rng.rand(n, n)
#     u, s, v = scipy.linalg.svd(np.dot(cov.T, cov))
#     cov = np.dot(np.dot(u, 1.0 + np.diag(rng.rand(n))), v)
#     delta = rng.normal(size=n)
#     ns = 100
#
#     with tempfile.TemporaryDirectory() as tmpdir:
#         rows = []
#         for i in range(n):
#             for j in range(n):
#                 rows.append((i, j, cov[i, j]))
#         df = pd.DataFrame.from_records(rows, columns=['i', 'j', 'cov'])
#
#         fname = os.path.join(tmpdir, 'cov.dat')
#         df.to_csv(fname, index=False)
#
#         lk = TDistLikelihood(covariance=fname, n=ns, data_vector=['a', 'b'])
#
#     assert lk.data_vector == ['a', 'b']
#     assert lk.n == ns
#     chi2 = np.dot(delta, np.dot(np.linalg.inv(cov), delta))
#     loglike = -0.5 * ns * np.log(1.0 + chi2 / (ns - 1.0))
#     assert np.allclose(loglike, lk(delta, np.zeros(n)))
