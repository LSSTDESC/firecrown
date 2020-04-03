"""Functions to test analysis algorithms."""


def parse_config(analysis):
    return {}


def compute_loglike(
        *,
        cosmo,
        parameters,
        data):
    return -0.5 * (parameters['x'] - 0.5)**2, None, None, None, None, {}


def write_stats(*, output_path, data, stats):
    """Write statistics to a file at `output_path`.

    Parameters
    ----------
    output_path : str
        The path to which to write the data.
    data : dict
        The output of `parse_config`.
    stats : object or other data
        Second output of `compute_loglike`.
    """
    pass
