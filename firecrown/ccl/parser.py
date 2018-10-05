import copy
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

import pyccl as ccl


def _parse_statistics(statistics):
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
        df = pd.read_csv(keys['data'])
        if keys['kind'] == 'cl':
            new_keys['l'] = df['l'].values.copy()
            new_keys['cl'] = df['cl'].values.copy()
        elif keys['kind'] in ['gg', 'gl', 'l+', 'l-']:
            new_keys['t'] = df['t'].values.copy()
            new_keys['xi'] = df['xi'].values.copy()

        stats[stat] = new_keys

    return stats


def _parse_sources(srcs):
    """Parse a source from the config.

    Each source expressed in YAML should have the form:

        ```YAML
        src0:
          kind: 'ClTracerLensing'
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
        del _src['nz_data']
        _src['kind'] = getattr(ccl, keys['kind'])
        df = pd.read_csv(keys['nz_data'])
        _z, _nz = df['z'].values.copy(), df['nz'].values.copy()
        _src['z_n'] = _z
        _src['n'] = _nz
        _src['pz_spline'] = Akima1DInterpolator(_z, _nz)
        sources[name] = _src
    return sources
