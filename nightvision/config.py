import yaml

from .sources import parse_ccl_source
from .likelihoods import parse_two_point


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

    analyses = list(
        set(list(data.keys())) -
        set(['sources', 'parameters', 'run_metadata']))
    for analysis in analyses:
        if analysis == 'two_point':
            data['two_point'] = parse_two_point(**config['two_point'])
        else:
            raise ValueError(
                "Analysis '%s' not recognized!" % (analysis))

    return config, data
