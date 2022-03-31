import numpy as np

from ..tdist import TdistLogLike


def test_likelihood_tdist_smoke(likelihood_test_data):
    nu = 25
    ll = TdistLogLike(data_vector=likelihood_test_data["data_vector"], nu=nu)
    ll.read(
        likelihood_test_data["sacc_data"],
        likelihood_test_data["sources"],
        likelihood_test_data["statistics"],
    )
    assert ll.data_vector == likelihood_test_data["data_vector"]

    delta = likelihood_test_data["delta"]
    data = likelihood_test_data["data"]
    theory = likelihood_test_data["theory"]
    cov = likelihood_test_data["cov"]
    chi2 = np.dot(delta, np.dot(np.linalg.inv(cov), delta))
    loglike = -0.5 * nu * np.log(1.0 + chi2 / (nu - 1.0))
    assert np.allclose(loglike, ll.compute(data, theory))

    dv = np.concatenate([data[v] for v in ll.data_vector])
    assert np.allclose(ll.assemble_data_vector(data), dv)


def test_likelihood_tdist_subset(likelihood_test_data):
    nu = 25
    ll = TdistLogLike(data_vector=["stat_src0_src0", "stat_src0_src1"], nu=nu)
    ll.read(
        likelihood_test_data["sacc_data"],
        likelihood_test_data["sources"],
        likelihood_test_data["statistics"],
    )
    assert ll.data_vector == ["stat_src0_src0", "stat_src0_src1"]

    delta = likelihood_test_data["delta"][0:4]
    data = likelihood_test_data["data"]
    theory = likelihood_test_data["theory"]
    cov = likelihood_test_data["cov"][0:4, 0:4]
    chi2 = np.dot(delta, np.dot(np.linalg.inv(cov), delta))
    loglike = -0.5 * nu * np.log(1.0 + chi2 / (nu - 1.0))
    assert np.allclose(loglike, ll.compute(data, theory))

    dv = np.concatenate([data[v] for v in ll.data_vector])
    assert np.allclose(ll.assemble_data_vector(data), dv)
