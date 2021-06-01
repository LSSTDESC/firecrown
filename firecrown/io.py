import os
import yaml

from .parser_constants import FIRECROWN_RESERVED_NAMES


def write_statistics(*, output_dir, data, statistics):
    """Write statistics to an output path.

    Parameters
    ----------
    output_dir : str
        The directory in which to write the statistics.
    data : dict
        The output of `parse_config`.
    statistics : dict
        Dictionary containing the output `stats` for each analysis.
    """
    _odir = os.path.join(output_dir, "statistics")
    os.makedirs(_odir, exist_ok=True)

    analyses = list(set(list(data.keys())) - set(FIRECROWN_RESERVED_NAMES))
    for analysis in analyses:
        _ana_odir = os.path.join(_odir, analysis)
        os.makedirs(_ana_odir, exist_ok=True)

        data[analysis]["write"](
            output_path=_ana_odir,
            data=data[analysis]["data"],
            stats=statistics[analysis],
        )

    with open(os.path.join(_odir, "parameters.yaml"), "w") as fp:
        yaml.dump(data["parameters"], fp)
