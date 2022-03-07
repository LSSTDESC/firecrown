import collections.abc
import copy
import importlib
from importlib import import_module as imp
import yaml
import jinja2

from .parser_constants import FIRECROWN_RESERVED_NAMES


def parse(config_or_filename):
    """Parse a configuration file.

    Parameters
    ----------
    config_or_filename : str or dict
        The config file to parse or an already parsed config dictionary.
        If a file, the file should be YAML formatted.

    Returns
    -------
    config : dict
        The raw config file as a dictionary.
    data : dict
        A dictionary containg each analyses key replaced with its
        corresponding data and function to compute the log-likelihood.
    """

    if not isinstance(config_or_filename, collections.abc.MutableMapping):
        with open(config_or_filename, "r") as fp:
            config_str = jinja2.Template(fp.read()).render()
        config = yaml.load(config_str, Loader=yaml.Loader)
        data = yaml.load(config_str, Loader=yaml.Loader)
    else:
        config = copy.deepcopy(config_or_filename)
        data = copy.deepcopy(config_or_filename)

    #print("+++++++++++++++++")
    #print(config)
    #print("+++++++++++++++++")
    #print(data)
    #print("+++++++++++++++++")
    params = {}
    for p, val in data["parameters"].items():
        if isinstance(val, list) and not isinstance(val, str):
            params[p] = val[1]
        else:
            params[p] = val
    data["parameters"] = params

    analyses = set(data.keys()) - set(FIRECROWN_RESERVED_NAMES)
    #print(analyses)
    #print("+++++++++++++++++")
    for analysis in analyses:
        #print(analysis)
        #print("+++++++++++++++++")
        new_keys = {}
        try:
            mod = imp(data[analysis]["module"])
        except Exception:
            #print("data, analysis ***************%s "%(data[analysis]))
            print(
                "Module '%s' for analysis '%s' cannot be imported!"
                % (data[analysis]["module"], analysis)
            )
            raise

        new_keys = {}
        if hasattr(mod, "parse_config"):
            #print("++++++++++++++")
            new_keys["data"] = getattr(mod, "parse_config")(data[analysis])
            new_keys["eval"] = getattr(mod, "compute_loglike")
            new_keys["write"] = getattr(mod, "write_stats")
        else:
            raise ValueError("Analsis '%s' could not be parsed!" % (analysis))

        data[analysis] = new_keys

    return config, data
