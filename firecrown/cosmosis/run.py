import pathlib
import os
import sys

# Locate the path to this directory
# For fiddly reasons that make a great deal of sense in an entirely
# different context to this one, we pass cosmosis a full path
# to the interface file
THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()
COSMOSIS_INTERFACE = str(THIS_DIRECTORY.joinpath('interface.py'))


def run_cosmosis(config, data):
    """Run CosmoSIS on the problem.

    This requires the following parameters 'sampler' section
    of the config:

      sampler - name of sampler to use, e.g. emcee, multinest, grid, ...
      output - name of file to save to


    Parameters
    ----------
    config : dict
        Configuration info, usually read directly from the YAML file
    data : dict
        The result of calling `firecrown.config.parse` on an input YAML
        config.

    """
    from cosmosis.main import run_cosmosis

    # Extract the bits of the config file that
    # cosmosis wants
    ini = _make_cosmosis_config(config['sampler'])
    values = _make_cosmosis_values(config['parameters'])
    pool = _make_parallel_pool(config['sampler'])
    pipeline = _make_cosmosis_pipeline(data, values, pool)

    # Actually run the thing
    run_cosmosis(None, pool=pool, ini=ini, pipeline=pipeline, values=values)

    if pool is not None:
        pool.close()


def _make_parallel_pool(config):
    """ Set up a parallel process pool.

    Parameters
    ----------

    config: dict
        Sampler configuration section of the input

    Will look for the 'mpi' command in the parallel section

    Returns
    -------

    pool: CosmoSIS MPIPool object
        parallel process pool
    """

    from cosmosis.runtime.mpi_pool import MPIPool

    # There is a reason to make the user actively
    # request to use MPI rather than just checking -
    # on many systems, including, importantly, NERSC,
    # trying to import MPI when not running under the
    # MPI environment will cause a crash
    use_mpi = config.get('mpi', False)

    if use_mpi:
        pool = MPIPool()
        if pool.size == 1:
            print("Have mpi set to True but one process.")
            print("I will ignore and run in serial mode.")
            pool = None
    else:
        pool = None
        print("Running in serial mode (one process).")

    return pool


def _make_cosmosis_pipeline(data, values, pool):
    """ Build a CosmoSIS pipeline.

    Parameters
    ----------

    data: dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function

    values: Inifile
        Cosmosis object representing the input parameter values

    pool: MPIPool or None
        If using MPI parallelism, a CosmoSIS pool object.

    Returns
    -------

    pipeline: CosmoSIS pipeline objects
        Instantiated pipeline ready to run.
    """
    from cosmosis.runtime.pipeline import LikelihoodPipeline
    from cosmosis.runtime.module import Module
    from cosmosis.runtime.utils import stdout_redirected

    # Lie to CosmoSIS about where it is installed.
    os.environ['COSMOSIS_SRC_DIR'] = '.'

    # Build the pipeline that evaluates the likelihood.
    # We avoid printing various bits of output info by silencing stdout on
    # worker nodes.
    if (pool is None) or pool.is_master():
        pipeline = LikelihoodPipeline(load=False, values=values,
                                      priors=None)
    else:
        with stdout_redirected():
            pipeline = LikelihoodPipeline(load=False, values=values,
                                          priors=None)

    sys.stdout.flush()
    # Set up a single cosmosis module, which will be the interface
    # file in the same directory as this one
    module = Module('firecrown', COSMOSIS_INTERFACE)
    module.setup_functions(data)
    pipeline.modules = [module]

    return pipeline


def _make_cosmosis_config(config):
    """ Extract a cosmosis configuration object from a config dict

    Parameters
    ----------
    config: dict
        Configuration dictionary of 'sampler' section of yaml

    Returns
    -------

    cosmosis_config: Inifile
        object to use to build cosmosis pipeline
    """
    from cosmosis.runtime.config import Inifile

    # Some general options
    sampler_name = config['sampler']
    output_file = config['output']
    debug = config.get('debug', False)
    quiet = config.get('quiet', False)
    root = ""  # Dummy value to stop cosmosis complaining

    # Passive-aggressive error message
    if sampler_name == 'fisher':
        raise ValueError("The Fisher matrix sampler "
                         "does not work since the refactor - sorry.")

    # Make into a pair dictionary with the right cosmosis sections
    cosmosis_options = {
        ("runtime", "root"): root,
        ("runtime", "sampler"): sampler_name,
        ("output", "filename"): output_file,
        ("pipeline", "debug"): str(debug),
        ("pipeline", "quiet"): str(quiet),
    }

    # Set all the sampler configuration options from the
    # appropriate section of the config (e.g., the "grid"
    # section if using the grid sampler, etc.)
    sampler_config = config.get(sampler_name, {})
    for key, val in sampler_config.items():
        cosmosis_options[(sampler_name, key)] = str(val)

    # Convert into cosmosis Inifile format.
    cosmosis_config = Inifile(None, override=cosmosis_options)

    return cosmosis_config


def _make_cosmosis_values(params):
    """ Extract a cosmosis values object from a config dict

    Parameters
    ----------
    params: dict
        Configuration dictionary of 'parameters' section of input yaml

    Returns
    -------

    cosmosis_values: Inifile
        object to use to build cosmosis parameter ranges/values
    """
    from cosmosis.runtime.config import Inifile

    values = {}
    for p, v in params.items():
        key = ('params', p)
        if isinstance(v, list):
            values[key] = ' '.join(str(x) for x in v)
        else:
            values[key] = str(v)

    cosmosis_values = Inifile(None, override=values)

    return cosmosis_values
