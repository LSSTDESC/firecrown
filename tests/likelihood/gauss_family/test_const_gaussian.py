"""Unit testsing for ConstGaussian
"""

import pytest
import numpy as np

from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import (
    RequiredParameters,
    DerivedParameterCollection,
)


def test_require_nonempty_statistics():
    with pytest.raises(ValueError):
        _ = ConstGaussian(statistics=[])


def test_get_cov_fails_before_read(trivial_stats):
    likelihood = ConstGaussian(statistics=trivial_stats)
    with pytest.raises(AssertionError):
        _ = likelihood.get_cov()


def test_get_cov_works_after_read(trivial_stats, sacc_data):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data)
    assert np.all(likelihood.get_cov() == np.diag([4.0, 9.0, 16.0]))


def test_chisquared(trivial_stats, sacc_data, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data)
    likelihood.update(trivial_params)
    assert likelihood.compute_chisq(ModelingTools()) == 2.0


def test_required_parameters(trivial_stats, sacc_data, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data)
    likelihood.update(trivial_params)
    expected_params = RequiredParameters(params_names=["mean"])
    assert likelihood.required_parameters() == expected_params


def test_derived_parameters(trivial_stats, sacc_data, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data)
    likelihood.update(trivial_params)
    expected_params = DerivedParameterCollection([])
    assert likelihood.get_derived_parameters() == expected_params


def test_reset(trivial_stats, sacc_data, trivial_params):
    likelihood = ConstGaussian(statistics=trivial_stats)
    likelihood.read(sacc_data)
    likelihood.update(trivial_params)
    assert not trivial_stats[0].computed_theory_vector
    assert likelihood.compute_chisq(ModelingTools()) == 2.0
    assert trivial_stats[0].computed_theory_vector
    likelihood.reset()
    assert not trivial_stats[0].computed_theory_vector
