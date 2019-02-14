"""Functions to test analysis algorithms."""


def parse_config(analysis):
    return {}


def compute_loglike(
        *,
        cosmo,
        parameters,
        data):
    return -0.5 * (parameters['x'] - 0.5)**2, {}
