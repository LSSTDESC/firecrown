import os
import yaml


def write_statistics(*, analysis_id, output_path, data, statistics):
    """Write statistics to an output path.

    Parameters
    ----------
    analysis_id : str
        A unique id for this analysis.
    output_path : str
        The path to which to write the run metdata.
    data : dict
        The output of `parse_config`.
    statistics : dict
        Dictionary containing the output `stats` for each analysis.
    """
    _opth = os.path.expandvars(os.path.expanduser(output_path))
    _odir = os.path.join(_opth, 'output_%s' % analysis_id, 'statistics')
    os.makedirs(_odir, exist_ok=True)

    analyses = list(
        set(list(data.keys())) -
        set(['parameters', 'cosmosis']))
    for analysis in analyses:
        _ana_odir = os.path.join(_odir, analysis)
        os.makedirs(_ana_odir, exist_ok=True)

        data[analysis]['write'](
            output_path=_ana_odir,
            data=data[analysis]['data'],
            stats=statistics[analysis])

    with open(os.path.join(_odir, 'parameters.yaml'), 'w') as fp:
        yaml.dump(data['parameters'], fp)


def write_analysis(analysis_id, output_path, chain_txt):
    """Write a chain to an output path.

    Parameters
    ----------
    analysis_id : str
        A unique id for this analysis.
    output_path : str
        The path to which to write the run metdata.
    chain_txt : str
        The cosmosis output file as a string.
    """
    _opth = os.path.expandvars(os.path.expanduser(output_path))
    _odir = os.path.join(_opth, 'output_%s' % analysis_id)
    os.makedirs(_odir, exist_ok=True)

    with open(os.path.join(_odir, 'chain.txt'), "w") as fp:
        fp.write(chain_txt)
