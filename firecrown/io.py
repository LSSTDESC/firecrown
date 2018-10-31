import os
import pandas as pd


def write_statistics(analysis_id, output_path, stats):
    """Write run metadata to an output path.

    Parameters
    ----------
    analysis_id : str
        A unique id for this analysis.
    output_path : str
        The path to which to write the run metdata.
    stats : dict
        Dictionary containing the statistics, each stored as numpy
        structured arrays.
    """
    _opth = os.path.expandvars(os.path.expanduser(output_path))
    _odir = os.path.join(_opth, 'output_%s' % analysis_id)
    os.makedirs(_odir)

    def _write_stats(_stats, pth):
        for key, value in _stats.items():
            if isinstance(value, dict):
                _pth = os.path.join(pth, key)
                os.makedirs(_pth)
                _write_stats(value, _pth)
            else:
                df = pd.DataFrame.from_records(value)
                df.to_csv(os.path.join(pth, key + '.csv'), index=False)

    _write_stats(stats, _odir)
