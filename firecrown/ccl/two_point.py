import os
import sacc

from .parser import (
    _parse_sources,
    _parse_systematics,
    _parse_two_point_statistics,
    _parse_likelihood)


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
    new_keys['systematics'] = _parse_systematics(analysis['systematics'])
    if 'likelihood' in analysis:
        new_keys['likelihood'] = _parse_likelihood(analysis['likelihood'])

    # read data if there is a sacc file
    if 'sacc_file' in analysis:
        sacc_data = sacc.Sacc.load_fits(
            os.path.expanduser(os.path.expandvars(analysis['sacc_file'])))
        for src in new_keys['sources']:
            new_keys['sources'][src].read(sacc_data)
        for stat in new_keys['statistics']:
            new_keys['statistics'][stat].read(sacc_data, new_keys['sources'])
        if 'likelihood' in new_keys:
            new_keys['likelihood'].read(
                sacc_data, new_keys['sources'], new_keys['statistics'])

    return new_keys


def compute_loglike(
        *,
        cosmo,
        parameters,
        data):
    """Compute the log-likelihood of 2pt data.

    Parameters
    ----------
    cosmo : a `pyccl.Cosmology` object
        A cosmology.
    parameters : dict
        Dictionary mapping parameters to their values.
    data : dict
        The output of `firecrown.ccl.two_point.parse_config`.

    Returns
    -------
    loglike : float
        The computed log-likelihood.
    stats : None
        Always None for this analysis.
    """

    for name, src in data['sources'].items():
        src.render(cosmo, parameters, systematics=data['systematics'])

    _data = {}
    _theory = {}
    for name, stat in data['statistics'].items():
        stat.compute(
            cosmo, parameters, data['sources'], systematics=data['systematics'])
        _data[name] = stat.measured_statistic_
        _theory[name] = stat.predicted_statistic_

    # compute the log-like
    if 'likelihood' in data:
        loglike = data['likelihood'].compute(_data, _theory)
    else:
        loglike = None

    return loglike, None


def write_stats(*, output_path, data, stats):
    """Write statistics to a file at `output_path`.

    Parameters
    ----------
    output_path : str
        The path to which to write the data.
    data : dict
        The output of `parse_config`.
    stats : object or other data
        Second output of `compute_loglike`. Always None for the two point
        analysis here.
    """
    meas_sacc, pred_sacc = build_sacc_data(data=data, stats=stats)
    meas_sacc.save_fits(
        os.path.join(output_path, 'sacc_measured.fits'), overwrite=True)
    pred_sacc.save_fits(
        os.path.join(output_path, 'sacc_predicted.fits'), overwrite=True)


def build_sacc_data(data, stats):
    """Build an SACC data file from a 2pt analysis computation.

    Parameters
    ----------
    data : dict
        The output of `parse_config`.
    stats : object or other data
        Second output of `compute_loglike`. Always None for the two point
        analysis here.

    Returns
    -------
    meas_sacc : sacc.Sacc
        The SACC data for the measured statistics.
    pred_sacc : sacc.Sacc
        The SACC data for the predicted statistics.
    """

    if 'likelihood' in data:
        names = data['likelihood'].data_vector
    else:
        names = list(data['statistics'].keys())

    base_sacc_data = sacc.Sacc()
    for name, src in data['sources'].items():
        base_sacc_data.add_tracer('NZ', src.sacc_tracer, src.z_orig, src.dndz_orig)

    datas = {}
    for attr in ['measured', 'predicted']:
        sacc_data = base_sacc_data.copy()

        for name in names:
            stat = data['statistics'][name]
            if stat.ccl_kind == 'cl':
                sacc_data.add_ell_cl(
                    stat.sacc_data_type,
                    *stat.sacc_tracers,
                    stat.ell_or_theta_,
                    getattr(stat, '%s_statistic_' % attr))
            else:
                sacc_data.add_theta_xi(
                    stat.sacc_data_type,
                    *stat.sacc_tracers,
                    stat.ell_or_theta_,
                    getattr(stat, '%s_statistic_' % attr))

        if 'likelihood' in data:
            sacc_data.add_covariance(data['likelihood'].cov)

        datas[attr] = sacc_data

    return datas['measured'], datas['predicted']
