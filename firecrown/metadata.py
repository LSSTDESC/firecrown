import os
import shutil
import datetime

import yaml
import pyccl

from ._version import __version__


def write_metadata(analysis_id, output_path, config_file):
    """Write run metadata to an output path.

    Parameters
    ----------
    analysis_id : str
        A unique id for this analysis.
    output_path : str
        The path to which to write the run metdata.
    config_file : str
        The path to the config file.
    """

    metadata = {
        'analysis_id': analysis_id,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'firecrown_version': __version__,
        'pyccl_version': pyccl.__version__}

    _opth = os.path.expandvars(os.path.expanduser(output_path))
    _odir = os.path.join(_opth, 'output_%s' % metadata['analysis_id'])
    os.makedirs(_odir)

    shutil.copyfile(config_file, os.path.join(_odir, 'config.yaml'))
    with open(os.path.join(_odir, 'metadata.yaml'), 'w') as fp:
        yaml.dump(metadata, fp, default_flow_style=False)
