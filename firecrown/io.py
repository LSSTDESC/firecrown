import os
import pandas as pd
import yaml


def write_statistics(*, analysis_id, output_dir, data, statistics):
    """Write statistics to an output path.

    Parameters
    ----------
    analysis_id : str
        A unique id for this analysis.
    output_dir : str or Path
        The directory in which which to write the statistics.
    data : dict
        The output of `parse_config`.
    statistics : dict
        Dictionary containing the output `stats` for each analysis.
    """
    _odir = os.path.join(output_dir, 'statistics')
    os.makedirs(_odir, exist_ok=True)

    analyses = list(
        set(list(data.keys())) -
        set(['parameters', 'cosmosis', 'emcee']))
    for analysis in analyses:
        _ana_odir = os.path.join(_odir, analysis)
        os.makedirs(_ana_odir, exist_ok=True)

        data[analysis]['write'](
            output_path=_ana_odir,
            data=data[analysis]['data'],
            stats=statistics[analysis])

    with open(os.path.join(_odir, 'parameters.yaml'), 'w') as fp:
        yaml.dump(data['parameters'], fp)


def write_analysis(analysis_id, output_path, chain):
    """Write a chain to an output path.

    Parameters
    ----------
    analysis_id : str
        A unique id for this analysis.
    output_path : str
        The path to which to write the run metdata.
    chain : numpy structured array
        The MCMC chain as a structured array.
    """
    _opth = os.path.expandvars(os.path.expanduser(output_path))
    _odir = os.path.join(_opth, 'output_%s' % analysis_id)
    os.makedirs(_odir, exist_ok=True)

    df = pd.DataFrame.from_records(chain)
    df.to_csv(os.path.join(_odir, 'analysis.csv'), index=False)
