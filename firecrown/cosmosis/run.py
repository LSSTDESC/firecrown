import os
import sys
import numbers
from ..cosmology import get_ccl_cosmology, RESERVED_CCL_PARAMS
from ..loglike import compute_loglike

try:
    import cosmosis
except ImportError:
    cosmosis = None


def run_cosmosis(config, data):
    """Run CosmoSIS on the problem.

    This requires the following parameters 'cosmosis' section
    of the config:

      sampler - name of sampler to use, e.g. emcee, multinest, grid, ...
      output - name of file to save to

      a section with the same name as the sampler, selecting options
      for that sampler.

    Parameters
    ----------
    config : dict
        Configuration info, usually read directly from the YAML file

    data : dict
        The result of calling `firecrown.config.parse` on an input YAML
        config.
    """

    if cosmosis is None:
        raise ImportError("CosmoSIS is not installed. "
                          "See readme for instructions on doing so.")

    # Extract the bits of the config file that
    # cosmosis wants
    ini = _make_cosmosis_params(config)
    values = _make_cosmosis_values(config)
    pool = _make_parallel_pool(config)
    priors = _make_cosmosis_priors(config)
    pipeline = _make_cosmosis_pipeline(data, ini, values, priors, pool)

    # Actually run the thing
    cosmosis.main.run_cosmosis(None, pool=pool, ini=ini,
                               pipeline=pipeline, values=values)

    if pool is not None:
        pool.close()


def _make_parallel_pool(config):
    """Set up a parallel process pool.

    Will look for the 'mpi' key in the cosmosis_config.

    Parameters
    ----------
    cosmosis_config: dict
        Sampler configuration section of the input

    Returns
    -------
    pool: CosmoSIS MPIPool object
        parallel process pool
    """
    cosmosis_config = config['cosmosis']

    # There is a reason to make the user actively
    # request to use MPI rather than just checking -
    # on many systems, including, importantly, NERSC,
    # trying to import MPI when not running under the
    # MPI environment will cause a crash
    use_mpi = cosmosis_config.get('mpi', False)

    if use_mpi:
        pool = cosmosis.MPIPool()
        if pool.size == 1:
            print("Have mpi=True, but only running a single process.")
            print("I will ignore and run in serial mode.")
            pool = None
    else:
        pool = None
        print("Running in serial mode (one process).")

    return pool


def _make_cosmosis_pipeline(data, ini, values, priors, pool):
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

    # Lie to CosmoSIS about where it is installed.
    os.environ['COSMOSIS_SRC_DIR'] = '.'

    # Build the pipeline that evaluates the likelihood.
    # We avoid printing various bits of output info by silencing stdout on
    # worker nodes.
    if (pool is None) or pool.is_master():
        pipeline = cosmosis.LikelihoodPipeline(load=False, values=values,
                                      priors=priors)
    else:
        with cosmosis.stdout_redirected():
            pipeline = cosmosis.LikelihoodPipeline(load=False, values=values,
                                          priors=priors)

    # Flush now to print out the master node's setup stdout
    # before printing the worker likelihoods
    sys.stdout.flush()

    # Set up a single cosmosis module, from the functions directly
    module = cosmosis.FunctionModule('firecrown', _setup, _execute)
    module.setup_functions((data, ini))
    pipeline.modules = [module]

    return pipeline


def _make_cosmosis_params(config):
    """Extract a cosmosis configuration object from a config dict

    Parameters
    ----------
    cosmosis_config: dict
        Configuration dictionary of 'cosmosis' section of yaml

    Returns
    -------
    cosmosis_params: Inifile
        object to use to build cosmosis pipeline
    """

    cosmosis_config = config['cosmosis']

    # Some general options
    sampler_name = cosmosis_config['sampler']
    output_file = cosmosis_config['output']
    debug = cosmosis_config.get('debug', False)
    quiet = cosmosis_config.get('quiet', False)
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
    # appropriate section of the cosmosis_config (e.g., the "grid"
    # section if using the grid sampler, etc.)
    sampler_config = cosmosis_config.get(sampler_name, {})
    for key, val in sampler_config.items():
        cosmosis_options[(sampler_name, key)] = str(val)

    # The string parameters in the yaml file parameters
    # can't go into cosmosis values, because that is for parameters
    # that might vary during a run, which string params will not.
    # Instead we put these in the parameter file
    for p, v in config['parameters'].items():
        if isinstance(v, str):
            cosmosis_options['firecrown', p] = v
            print(f"Setting string parameter {p} = {v}")

    # Convert into cosmosis Inifile format.
    cosmosis_params = cosmosis.Inifile(None, override=cosmosis_options)

    return cosmosis_params


def _make_cosmosis_values(config):
    """Extract a cosmosis values object from a config dict

    Parameters
    ----------
    params: dict
        Configuration dictionary of 'parameters' section of input yaml

    Returns
    -------
    cosmosis_values: Inifile
        object to use to build cosmosis parameter ranges/values
    """
    params = config['parameters']
    varied_params = config['cosmosis']['parameters']

    # copy all the parameters into the cosmosis config structure
    values = {}

    # First set all the numeric parameters, fixed and varied.
    # We will override the varied ones in a moment
    for p, v in params.items():
        if isinstance(v, numbers.Number):
            values['params', p] = str(v)

    # Now override the varied parameters
    for p, v in varied_params.items():
        v = ' '.join(str(x) for x in v)
        values['params', p] = v

    return cosmosis.Inifile(None, override=values)


def _make_cosmosis_priors(config):
    """Make a cosmosis priors ini file
    
    """
    P = {}
    for name, p in config['priors'].items():
        # FireCrown exposes any scipy distribtion as a prior.
        # CosmoSIS only exposes three of these right now (plus
        # a couple of others that scipy doesn't support), but
        # these are by far the most common.
        
        # This is a key used by other FireCrown tools        
        if name == 'module':
            continue

        # The
        kind = p['kind']
        loc = p['loc']
        scale = p['scale']
        # Flat
        if kind == 'uniform':
            upper = loc + scale
            pr = f'uniform {loc} {upper}'
        # Exponential, only with loc = 0
        elif kind == 'expon':
            # This is not currently in CosmoSIS.  It's not hard to add,
            # and if there is demand Joe can add it.
            if loc != 0:
                raise ValueError("CosmoSIS does not currently support exponential "
                                 "priors with non-zero 'loc'.  If you need this please "
                                 "open an issue")
            pr = f'exp {scale}'
        # Gaussian.
        elif kind == 'norm':
            pr = f'norm {loc} {scale}'
        else:
            raise ValueError(f"CosmoSIS does not know how to use the prior kind {kind}")
        # Put these all in a dictionary
        P['params', name] = pr

    return cosmosis.Inifile(None, override=P)


def _setup(data):
    # In most cosmosis modules this function
    # is a bit more complicated!  We are hacking it
    # a bit here - normally the input to this function
    # is the configuration information for the module.
    return data


def _execute(block, data):
    data, ini = data
    # Calculate the firecrown likelihood as a module
    # This function, which isn't designed for end users,
    # is the main connection between cosmosis and firecrown.
    # CosmoSIS builds the block, and passes it to us here.
    # The block contains all the sample parameters.

    # Create CCL cosmology
    ccl_values = {}

    for p in RESERVED_CCL_PARAMS:
        # First look in the block
        if block.has_value('params', p):
            ccl_values[p] = block['params', p]
        # Then in the ini file, for string params
        elif ini.has_option('firecrown', p):
            ccl_values[p] = ini.get('firecrown', p)

    cosmo = get_ccl_cosmology(ccl_values)

    # Put all the parameters in the data dictionary,
    # both CCL-related and others, like nuisance params.
    all_params = data['parameters'].keys()
    for p in all_params:
        # string parameters are excluded here, and potentially others
        if block.has_value('params', p):
            data['parameters'][p] = block['params', p]

    # Call out to the log likelihood
    loglike, stats = compute_loglike(cosmo=cosmo, data=data)

    # Send result back to cosmosis
    block['likelihoods', 'firecrown_like'] = loglike

    # Unless in quiet mode, print out what we have done
    if not data['cosmosis'].get("quiet", True):
        print("params = {}".format(data['parameters']))
        print(f"loglike = {loglike}\n", flush=True)

    # Signal success.  An exception anywhere above will
    # be converted to a -inf likelihood by default.
    return 0
