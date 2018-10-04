import copy
import yaml

from . import sources
from . import analyses
from . import systematics
from . import likelihoods


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
            sources[name] = parse_ccl_source(**keys)
        else:
            raise ValueError(
                "Source type '%s' not recognized for source '%s'!" % (
                    name, keys['type']))
    data['sources'] = sources

    _analyses = list(
        set(list(data.keys())) -
        set(['sources', 'parameters', 'run_metadata']))
    for analysis in _analyses:
        new_keys = {}

        if hasattr(likelihoods, data[analysis]['likelihood']['kind']):
            _cfg = copy.deepcop(data[analysis]['likelihood'])
            _cfg.pop('kind')
            new_keys['likelihood'] = getattr(
                    likelihoods, data[analysis]['likelihood'][''])(**_cfg)
        else:
            raise ValueError(
                "Likelihood '%s' not recognized for source "
                "'%s'!" % (data[analysis]['likelihood']['kind'], analysis))

        if analysis == 'two_point':
            new_keys['statistics'] = parse_two_point(
                config[analysis]['statistics'])
        else:
            raise ValueError(
                "Analysis '%s' not recognized!" % (analysis))

        data[analysis] = new_keys

    return config, data
