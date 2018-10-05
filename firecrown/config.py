import importlib
import yaml


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
        A dictionary containg each analyses key replaced with its
        corresponding data and function to compute the loglikelihood.
    """

    with open(filename, 'r') as fp:
        config = yaml.load(fp)

    with open(filename, 'r') as fp:
        data = yaml.load(fp)

    analyses = list(
        set(list(data.keys())) -
        set(['parameters', 'run_metadata']))
    for analysis in analyses:
        new_keys = {}

        try:
            mod = importlib.import_module(data[analysis]['module'])
        except Exception:
            print("Module '%s' for analysis '%s' cannot be imported!" % (
                data[analysis]['module'], analysis))
            raise

        new_keys[analysis] = {}
        if hasattr(mod, 'parse_config'):
            new_keys[analysis]['data'] = getattr(
                mod, 'parse_config')(data[analysis])
            new_keys[analysis]['eval'] = getattr(
                mod, 'compute_loglike')
        else:
            raise ValueError("Analsis '%s' could not be parsed!" % (analysis))

        data[analysis] = new_keys

    return config, data
