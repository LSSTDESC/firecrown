import copy

from .statistics import TwoPointStatistic
from . import sources as firecrown_ccl_sources
from . import likelihoods as firecrown_ccl_likelihoods
from . import systematics as firecrown_ccl_systematics


def _parse_systematics(systematics):
    """Parse the systematics.

    Expressed as YAML it should have the form

    ```YAML
    wl_m0:
      kind: MultiplicateShearBias:
      m: 'm0'
    ```
    """
    # extract sources
    syss = {}
    for name, keys in systematics.items():
        _src = copy.deepcopy(keys)
        kind = _src.pop('kind')
        syss[name] = getattr(firecrown_ccl_systematics, kind)(**_src)
    return syss


def _parse_two_point_statistics(statistics):
    """Parse two-point statistics from the config.

    Each stat expressed in YAML should have the form:

        ```YAML
        cl_src0_src0:
          sources: ['src0', 'src0']
          kind: 'cl'
          data: ./data/cl00.csv
          systematics:
            ...
        ```

    The kind can be any of the CCL 2pt function types ('gg', 'gl', 'l+',
    'l-') or 'cl' for Fourier space statistics.
    """
    stats = {}
    for stat, keys in statistics.items():
        new_keys = copy.deepcopy(keys)
        stats[stat] = TwoPointStatistic(**new_keys)
    return stats


def _parse_sources(srcs):
    """Parse sources from the config.

    Each source expressed in YAML should have the form:

        ```YAML
        src0:
          kind: 'WLSource'
          nz_data: ./data/cl00.csv
          has_intrinsic_alignment: False
          systematics:
            ...
        ```

    The kind can be any of the CCL 2pt function types ('gg', 'gl', 'l+',
    'l-') or 'cl' for Fourier space statistics.
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
          data: ./data/cov.csv
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
