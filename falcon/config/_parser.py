import yaml

__all__ = ['parse']


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

    return config, config
