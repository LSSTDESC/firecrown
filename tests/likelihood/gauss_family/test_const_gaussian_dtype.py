"""Public Gaussian workflow checks for float64 covariance and vector storage."""

import numpy as np
import numpy.typing as npt
import pytest
import sacc

from firecrown.likelihood._gaussian import ConstGaussian
from firecrown.likelihood._statistic import TrivialStatistic
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import ParamsMap


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("read", [True, False])
def test_gaussian_real_ingress(
    dtype: type[np.int32] | type[np.float32] | type[np.float64], read: bool
) -> None:
    """Preserve values, ownership, likelihoods and seeded realizations.

    :param dtype: The physical covariance and observation dtype at ingress.
    :param read: Use SACC read or ready construction.
    """
    data = sacc.Sacc()
    for value in np.array([1, 4, -3], dtype=dtype):
        data.add_data_point("count", (), value)
    covariance = np.diag(np.array([4, 9, 16], dtype=dtype))
    data.add_covariance(covariance)
    statistic = TrivialStatistic()
    if read:
        likelihood = ConstGaussian([statistic])
        likelihood.read(data)
    else:
        statistic.read(data)
        ready_likelihood = ConstGaussian.create_ready([statistic], covariance)
        assert isinstance(ready_likelihood, ConstGaussian)
        likelihood = ready_likelihood
    covariance[0, 0] = 100
    assert likelihood.get_cov().dtype == np.float64
    np.testing.assert_array_equal(likelihood.get_cov(), np.diag([4.0, 9.0, 16.0]))
    observed = likelihood.get_data_vector()
    assert observed.dtype == np.float64
    np.testing.assert_array_equal(observed, [1.0, 4.0, -3.0])
    observed[0] = 100
    assert likelihood.get_data_vector()[0] == 1
    assert np.issubdtype(likelihood.get_sacc_indices().dtype, np.integer)
    likelihood.update(ParamsMap(mean=1.0))
    tools = ModelingTools()
    assert likelihood.compute_loglike(tools) == -1.0
    theory = likelihood.get_theory_vector()
    assert theory.dtype == np.float64
    np.testing.assert_array_equal(theory, [1.0, 1.0, 1.0])
    theory[0] = 100
    assert likelihood.get_theory_vector()[0] == 1
    noiseless = likelihood.make_realization(data, add_noise=False)
    np.testing.assert_array_equal(noiseless.mean, [1.0, 1.0, 1.0])
    state = np.random.get_state()
    try:
        np.random.seed(123)
        realization = likelihood.make_realization_vector()
    finally:
        np.random.set_state(state)
    assert realization.dtype == np.float64
    np.testing.assert_allclose(
        realization,
        [-1.1712612066011223, 3.9920363397507574, 2.131913992207968],
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "covariance",
    [
        np.eye(3, dtype=np.bool_),
        np.eye(3, dtype=np.complex128),
        np.eye(3).astype(str),
        np.eye(3, dtype=object),
        np.zeros((3, 3), dtype=[("value", np.float64)]),
        np.zeros((3, 3), dtype="datetime64[D]"),
        np.ma.array(np.eye(3), mask=False),
        np.ma.array(np.eye(3), mask=np.eye(3, dtype=np.bool_)),
    ],
)
def test_gaussian_rejects_unsupported_covariance(
    covariance: npt.NDArray[np.generic], sacc_data_for_trivial_stat: sacc.Sacc
) -> None:
    """Reject invalid physical covariance instead of silently dropping information.

    :param covariance: Deliberately invalid physical covariance storage.
    :param sacc_data_for_trivial_stat: Observations used to ready the statistic.
    """
    statistic = TrivialStatistic()
    statistic.read(sacc_data_for_trivial_stat)
    with pytest.raises(TypeError, match="real integer or floating|masked"):
        ConstGaussian.create_ready(
            [statistic],
            covariance,  # type: ignore[arg-type]  # Test invalid runtime ingress.
        )
