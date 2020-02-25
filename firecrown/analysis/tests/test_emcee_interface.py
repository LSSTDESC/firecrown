import numpy as np

import pytest

from ..emcee_interface import run_emcee
from ..test_loglike import compute_loglike


@pytest.mark.parametrize('n_walkers', [10, 'max(10.2, 5.0)', '9 + n_dims'])
def test_run_emcee(n_walkers):
    config = {
        'parameters': dict(
            Omega_c=0.27,
            Omega_b=0.045,
            Omega_k=0.0,
            w0=-1.0,
            wa=0.0,
            sigma8=0.8,
            n_s=0.96,
            h=0.67,
            x=0.3),
        'emcee_test': {'module': 'firecrown.analysis.test_loglike'}}
    data = {
        'parameters': dict(
            Omega_c=0.27,
            Omega_b=0.045,
            Omega_k=0.0,
            w0=-1.0,
            wa=0.0,
            sigma8=0.8,
            n_s=0.96,
            h=0.67,
            x=0.3),
        'emcee_test': {'eval': compute_loglike, 'data': None}}
    parameters = ['x']
    n_steps = 1000

    stats, chain = run_emcee(
        config, data,
        parameters=parameters,
        n_steps=n_steps,
        n_walkers=10)

    msk = chain['mcmc_step'] >= 600

    mn = np.mean(chain['x'][msk])
    sd = np.std(chain['x'][msk])
    assert np.abs((mn - 0.5) / sd) <= 1
    assert np.max(chain['emcee_walker']) == 9
    assert np.max(chain['mcmc_step']) == 999
    assert type(data['parameters']['x']) == float
