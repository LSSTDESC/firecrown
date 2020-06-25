import os
import sacc

from .parser import (
    _parse_sources,
    _parse_systematics,
    _parse_two_point_statistics,
    _parse_likelihood)

from ._ccl import compute_loglike, write_stats  # noqa


def parse_config(analysis):
    """Parse a nx2pt analysis.

    Parameters
    ----------
    analysis : dict
        Dictionary containing the Nx2pt analysis.

    Returns
    -------
    data : dict
        Dictionary holding all of the data needed for a Nx2pt analysis.
    """
    new_keys = {}
    new_keys['sources'] = _parse_sources(analysis['sources'])
    new_keys['statistics'] = _parse_two_point_statistics(analysis['statistics'])
    if 'systematics' in analysis:
        new_keys['systematics'] = _parse_systematics(analysis['systematics'])
    else:
        new_keys['systematics'] = {}
    if 'likelihood' in analysis:
        new_keys['likelihood'] = _parse_likelihood(analysis['likelihood'])

    # read data if there is a sacc file
    if 'sacc_data' in analysis:
        if isinstance(analysis["sacc_data"], sacc.Sacc):
            sacc_data = analysis["sacc_data"]
        else:
            sacc_data = sacc.Sacc.load_fits(
                os.path.expanduser(os.path.expandvars(analysis['sacc_data'])))

        for src in new_keys['sources']:
            new_keys['sources'][src].read(sacc_data)
        for stat in new_keys['statistics']:
            new_keys['statistics'][stat].read(sacc_data, new_keys['sources'])
        if 'likelihood' in new_keys:
            new_keys['likelihood'].read(
                sacc_data, new_keys['sources'], new_keys['statistics'])

    return new_keys
