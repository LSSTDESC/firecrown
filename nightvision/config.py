import yaml
from scipy.interpolate import Akima1DInterpolator

import pandas as pd
import numpy as np

import pyccl as ccl

from .sources import build_ccl_source


def _parse_ccl_source(keys):
    new_keys = {}
    new_keys['kind'] = getattr(ccl, keys.pop('kind'))
    df = pd.read_csv(keys.pop('data'))
    _z, _nz = df['z'].values.copy(), df['nz'].values.copy()
    new_keys['z'] = _z
    new_keys['n'] = _nz
    new_keys['pz_spline'] = Akima1DInterpolator(_z, _nz)
    new_keys['build_func'] = build_ccl_source
    new_keys.update(keys)
    return new_keys


def _parse_two_point_statistic(keys):
    new_keys = {}
    new_keys.update(keys)
    df = pd.read_csv(keys['data'])
    if keys['kind'] == 'cl':
        new_keys['l'] = df['l'].values.copy()
        new_keys['cl'] = df['cl'].values.copy()
    elif keys['kind'] in ['gg', 'gl', 'l+', 'l+']:
        new_keys['t'] = df['t'].values.copy()
        new_keys['xi'] = df['xi'].values.copy()

    return new_keys


def _parse_two_point_likelihood(keys):
    new_keys = {}
    df = pd.read_csv(keys.pop('data'))
    dim = max(np.max(df['i']), np.max(df['j'])) + 1
    cov = np.zeros((dim, dim))
    cov[df['i'].values, df['j'].values] = df['cov'].values
    new_keys['cov'] = cov
    new_keys['L'] = np.linalg.cholesky(cov)
    new_keys.update(keys)
    return new_keys


def parse(filename):
    """Parse a configuration file.

    Parameters
    ----------
    filename : str
        The config file to parse. Should be YAML formatted.

    Returns
    -------
    config: dict
        The raw config file as a dictionary.
    data : dict
        A dictionary containg each config file key replaced with its
        corresponding data structure.
    """

    with open(filename, 'r') as fp:
        config = yaml.load(fp)

    with open(filename, 'r') as fp:
        data = yaml.load(fp)

    # extract sources
    sources = {}
    for name, keys in data.get('sources', {}).items():
        if keys['kind'] in ['ClTracerLensing', 'ClTracerNumberCounts']:
            sources[name] = _parse_ccl_source(keys)
        else:
            raise ValueError(
                "Source type '%s' not recognized for source '%s'!" % (
                    name, keys['type']))
    data['sources'] = sources

    for analysis in data['analyses']:
        if analysis == 'two_point':
            data['two_point']['likelihood'] = _parse_two_point_likelihood(
                config['two_point']['likelihood'])
            stats = {}
            for stat, keys in config['two_point']['statistics'].items():
                stats[stat] = _parse_two_point_statistic(keys)
            data['two_point']['statistics'] = stats
        else:
            raise ValueError(
                "Analysis '%s' not recognized!" % (analysis))

    return config, data
