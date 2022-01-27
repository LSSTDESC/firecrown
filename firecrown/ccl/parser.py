import copy
import importlib

from . import statistics as firecrown_ccl_statistics
from . import sources as firecrown_ccl_sources
from . import likelihoods as firecrown_ccl_likelihoods
from . import systematics as firecrown_ccl_systematics


def _parse_systematics(systematics):
    """Parse the systematics.

    Expressed as YAML it should have the form

    ```YAML
    wl_m0:
      kind: MultiplicateShearBias
      m: 'm0'
    ```
    """
    # extract sources
    syss = {}
    for name, keys in systematics.items():
        _src = copy.deepcopy(keys)
        kind = _src.pop('kind')
        try:
            syss[name] = getattr(firecrown_ccl_systematics, kind)(**_src)
        except AttributeError:
            # this must in another module, try an import
            items = kind.split('.')
            kind = items[-1]
            mod = ".".join(items[:-1])
            mod = importlib.import_module(mod)
            syss[name] = getattr(mod, kind)(**_src)

    return syss


def _parse_two_point_statistics(statistics):
    """Parse two-point statistics from the config.

    Each stat expressed in YAML should have the form:

        ```YAML
        cl_src0_src0:
          sources: ['src0', 'src0']
          sacc_data_type: 'galaxy_density_xi'  # a SACC type that maps to a CCL
                                               # correlation function kind or
                                               # power spectrum
          systematics:
            ...
        ```
    """
    stats = {}
    for stat, keys in statistics.items():
        new_keys = copy.deepcopy(keys)
        stats[stat] = firecrown_ccl_statistics.TwoPointStatistic(**new_keys)
    return stats


def _parse_statistics(statistics):
    """Parse statistics from the config.

    Each stat expressed in YAML should have the form:

        ```YAML
        cl_src0_src0:
          kind: TwoPointStatistic
            ...
        ```
    """
    stats = {}
    for stat, keys in statistics.items():
        _stat = copy.deepcopy(keys)
        kind = _stat.pop('kind')
        stats[stat] = getattr(firecrown_ccl_statistics, kind)(**_stat)
    return stats


def _parse_sources(srcs):
    """Parse sources from the config.

    Each source expressed in YAML should have the form:

        ```YAML
        src0:
          kind: 'WLSource'
          sacc_tracer: 'bin_0'  # name of the tracer in the SACC file goes here
          systematics:
            ...
        ```

    The kind can be on of {'WLSource', 'NumberCountsSource'}.
    """
    # extract sources
    sources = {}
    for name, keys in srcs.items():
        _src = copy.deepcopy(keys)
        kind = _src.pop('kind')
        sources[name] = getattr(firecrown_ccl_sources, kind)(**_src)
    return sources


def _parse_likelihood(likelihood):
    """Parse the likelihood from the config.

    Expressed in YAML it should have the form:

        ```YAML
        likelihood:
          kind: 'ConstGaussianLogLike'
          data_vector:
            - stat1
            - stat2
            ...
        ```
    The argument to this function is the dictionary containg the keys parsed
    from the YAML.
    """
    _lk = copy.deepcopy(likelihood)
    kind = _lk.pop('kind')
    return getattr(firecrown_ccl_likelihoods, kind)(**_lk)
