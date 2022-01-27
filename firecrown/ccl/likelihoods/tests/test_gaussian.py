import numpy as np

from ..gaussian import ConstGaussianLogLike


def test_likelihood_gaussian_smoke(likelihood_test_data):
    ll = ConstGaussianLogLike(data_vector=likelihood_test_data['data_vector'])
    ll.read(
        likelihood_test_data['sacc_data'],
        likelihood_test_data['sources'],
        likelihood_test_data['statistics'])
    assert ll.data_vector == likelihood_test_data['data_vector']

    delta = likelihood_test_data['delta']
    data = likelihood_test_data['data']
    theory = likelihood_test_data['theory']
    cov = likelihood_test_data['cov']
    loglike = -0.5 * np.dot(delta, np.dot(np.linalg.inv(cov), delta))
    assert np.allclose(loglike, ll.compute(data, theory))

    dv = np.concatenate([data[v] for v in ll.data_vector])
    assert np.allclose(ll.assemble_data_vector(data), dv)


def test_likelihood_gaussian_subset(likelihood_test_data):
    ll = ConstGaussianLogLike(data_vector=["stat_src0_src0", "stat_src0_src1"])
    ll.read(
        likelihood_test_data['sacc_data'],
        likelihood_test_data['sources'],
        likelihood_test_data['statistics'])
    assert ll.data_vector == ["stat_src0_src0", "stat_src0_src1"]

    delta = likelihood_test_data['delta'][0:4]
    data = likelihood_test_data['data']
    theory = likelihood_test_data['theory']
    cov = likelihood_test_data['cov'][0:4, 0:4]
    loglike = -0.5 * np.dot(delta, np.dot(np.linalg.inv(cov), delta))
    assert np.allclose(loglike, ll.compute(data, theory))

    dv = np.concatenate([data[v] for v in ll.data_vector])
    assert np.allclose(ll.assemble_data_vector(data), dv)
