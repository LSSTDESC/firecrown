import numpy as np
import scipy.linalg
import pytest

import sacc


class DummyThing(object):
    pass


@pytest.fixture
def likelihood_test_data():
    rng = np.random.RandomState(seed=10)

    sacc_data = sacc.Sacc()

    sacc_data.add_tracer("NZ", "trc0", rng.uniform(size=10), rng.uniform(size=10))
    sacc_data.add_tracer("NZ", "trc1", rng.uniform(size=10), rng.uniform(size=10))
    sacc_data.add_tracer("NZ", "trc2", rng.uniform(size=10), rng.uniform(size=10))

    sacc_data.add_ell_cl(
        "galaxy_density_cl", "trc0", "trc0", rng.uniform(size=2), rng.uniform(size=2)
    )
    sacc_data.add_ell_cl(
        "galaxy_density_cl", "trc0", "trc1", rng.uniform(size=2), rng.uniform(size=2)
    )
    sacc_data.add_ell_cl(
        "galaxy_density_cl", "trc0", "trc2", rng.uniform(size=2), rng.uniform(size=2)
    )
    sacc_data.add_ell_cl(
        "galaxy_density_cl", "trc1", "trc1", rng.uniform(size=2), rng.uniform(size=2)
    )
    sacc_data.add_ell_cl(
        "galaxy_density_cl", "trc1", "trc2", rng.uniform(size=2), rng.uniform(size=2)
    )
    sacc_data.add_ell_cl(
        "galaxy_density_cl", "trc2", "trc2", rng.uniform(size=2), rng.uniform(size=2)
    )

    # code to make a random positive semi-definite, symmetric matrix
    n = 12  # 6 * 2
    cov = rng.rand(n, n)
    u, s, v = scipy.linalg.svd(np.dot(cov.T, cov))
    cov = np.dot(np.dot(u, 1.0 + np.diag(rng.rand(n))), v)

    sacc_data.add_covariance(cov)

    sources = {}
    statistics = {}
    data = {}
    theory = {}
    data_vector = []
    for i in range(3):
        srci = "src%d" % i
        trci = "trc%d" % i
        sources[srci] = DummyThing()
        sources[srci].sacc_tracer = "trc%d" % i
        for j in range(i, 3):
            srcj = "src%d" % j
            trcj = "trc%d" % j
            sname = "stat_%s_%s" % (srci, srcj)
            statistics[sname] = DummyThing()
            statistics[sname].sacc_data_type = "galaxy_density_cl"
            statistics[sname].sacc_tracers = (trci, trcj)
            statistics[sname].sacc_inds = sacc_data.indices(
                statistics[sname].sacc_data_type, statistics[sname].sacc_tracers
            )
            theory[sname] = np.zeros(2)
            data[sname] = sacc_data.get_ell_cl("galaxy_density_cl", trci, trcj)[1]
            data_vector.append(sname)

    delta = sacc_data.mean.copy()

    return {
        "sacc_data": sacc_data,
        "cov": cov,
        "sources": sources,
        "statistics": statistics,
        "delta": delta,
        "data": data,
        "theory": theory,
        "data_vector": data_vector,
    }
