import os
import shutil
import datetime

import yaml
import pyccl

from ._version import __version__


def write_metadata(analysis_id, output_dir, config_file):
    """Write run metadata to an output path.

    Parameters
    ----------
    analysis_id : str
        A unique id for this analysis.
    output_dir : str or Path
        The directory in which to write metadata
    config_file : str
        The path to the config file.
    """

    metadata = {
        'analysis_id': analysis_id,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'firecrown_version': __version__,
        'pyccl_version': pyccl.__version__}

    # Copy configuration file into output
    shutil.copyfile(config_file,
                    os.path.join(output_dir, 'config.yaml'))

    # Save any metadata
    with open(os.path.join(output_dir, 'metadata.yaml'), 'w') as fp:
        yaml.dump(metadata, fp, default_flow_style=False)
