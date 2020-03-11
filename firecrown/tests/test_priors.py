import numpy as np

from ..priors import parse_config, compute_loglike


def test_priors():
    analysis = {
        'mu': {'kind': 'norm', 'loc': 0.3, 'scale': 1.0},
        'h': {'kind': 'norm', 'loc': 0.1, 'scale': 2.0},
        'v': {'kind': 'norm', 'loc': -0.3, 'scale': 4.0}}
    parameters = {'mu': 0.0, 'h': -0.1}

    data = parse_config(analysis)
    assert data == analysis

    loglike, stats = compute_loglike(
        cosmo=None, parameters=parameters, data=data)

    assert loglike == (
        -0.5 * ((0.3) / 1.0)**2 - 0.5 * np.log(2.0 * np.pi) +
        -0.5 * ((-0.1 - 0.1) / 2.0)**2 - 0.5 * np.log(2.0 * np.pi) - np.log(2))
    assert stats == {}
