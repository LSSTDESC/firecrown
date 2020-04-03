import os
import sys
import numbers
from ..cosmology import get_ccl_cosmology, RESERVED_CCL_PARAMS
from ..loglike import compute_loglike
from ..parser_constants import FIRECROWN_RESERVED_NAMES
import numpy as np

import cosmosis

# these keys are ignored by cosmosis
RESERVED_NAMES_COSMOSIS = FIRECROWN_RESERVED_NAMES + ['priors']


def run_cosmosis(config, data, output_dir):
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
    output_dir : pathlib.Path
        Directory in which to put output.
    """

    # Extract the bits of the config file that
    # cosmosis wants
    ini = _make_cosmosis_params(config, output_dir)
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

    Will look for the 'mpi' key in the config cosmosis section.

    Parameters
    ----------
    config: dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function

    Returns
    -------
    pool : CosmoSIS MPIPool object
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
    data : dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function
    ini : Inifile
        Cosmosis object representing the main input parameter file
    values : Inifile
        Cosmosis object representing the input parameter values
    pool : MPIPool or None
        If using MPI parallelism, a CosmoSIS pool object.

    Returns
    -------
    pipeline : CosmoSIS pipeline objects
        Instantiated pipeline ready to run.
    """

    # Lie to CosmoSIS about where it is installed.
    os.environ['COSMOSIS_SRC_DIR'] = '.'

    # Build the pipeline that evaluates the likelihood.
    # We avoid printing various bits of output info by silencing stdout on
    # worker nodes.
    if (pool is None) or pool.is_master():
        pipeline = cosmosis.LikelihoodPipeline(ini, load=False, values=values,
                                               priors=priors)
    else:
        with cosmosis.stdout_redirected():
            pipeline = cosmosis.LikelihoodPipeline(ini, load=False, values=values,
                                                   priors=priors)

    # Flush now to print out the master node's setup stdout
    # before printing the worker likelihoods
    sys.stdout.flush()

    # Set up a single cosmosis module, from the functions directly
    module = cosmosis.FunctionModule('firecrown', _setup, _execute)
    module.setup_functions((data, ini))
    pipeline.modules = [module]

    return pipeline


def _make_cosmosis_params(config, output_dir):
    """Extract a cosmosis configuration object from a config dict

    Parameters
    ----------
    config : dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function
    output_dir : pathlib.Path
        Directory to put output into.

    Returns
    -------
    cosmosis_params : Inifile
        object to use to build cosmosis pipeline
    """

    cosmosis_config = config['cosmosis']

    # Some general options
    sampler_names = cosmosis_config['sampler']
    output_file = str(output_dir / 'chain.txt')
    debug = cosmosis_config.get('debug', False)
    quiet = cosmosis_config.get('quiet', False)
    root = ""  # Dummy value to stop cosmosis complaining

    # Make into a pair dictionary with the right cosmosis sections
    cosmosis_options = {
        ("runtime", "root"): root,
        ("runtime", "sampler"): sampler_names,
        ("output", "filename"): output_file,
        ("pipeline", "debug"): str(debug),
        ("pipeline", "quiet"): str(quiet),
    }

    # Set all the sampler configuration options from the
    # appropriate section of the cosmosis_config (e.g., the "grid"
    # section if using the grid sampler, etc.)
    for sampler_name in sampler_names.split():
        sampler_config = cosmosis_config.get(sampler_name, {})
        for key, val in sampler_config.items():
            cosmosis_options[sampler_name, key] = str(val)

    # Override options that involve the user-specified
    # output paths to put everything in the one directory
    overridden_options = [
        ('maxlike', 'output_ini', 'output.ini'),
        ('maxlike', 'output_cov', 'covmat.txt'),
        ('multinest', 'multinest_outfile_root', 'multinest'),
        ('gridmax', 'output_ini', 'maxlike.ini'),
        ('minuit', 'output_ini', 'maxlike.ini'),
        ('minuit', 'save_cov', 'covmat.txt'),
        ('pmaxlike', 'output_ini', 'maxlike.ini'),
        ('pmaxlike', 'output_covmat', 'covmat.txt'),
        ('polychord', 'polychord_outfile_root', 'polychord'),
        ('polychord', 'base_dir', ''),
    ]

    # Apply these overrides
    for section, key, value in overridden_options:
        # To avoid too much noise in headers, only
        # copy over sections for samplers we're actually
        # using
        if section not in sampler_names:
            continue
        full_value = output_dir / value
        # Only warn user if they tried to set this already
        if (section, key) in cosmosis_options:
            sys.stderr.write(f"NOTE: Overriding option {section}/{key}"
                             f" to {full_value}")
        cosmosis_options[section, key] = str(full_value)

    # These options are not enabled by default, because they can
    # produce large output files.  So we only override them if
    # they are already set
    optional_overrides = [
        ('aprior', 'save', 'save'),
        ('grid', 'save', 'save'),
        ('list', 'save', 'save'),
        ('minuit', 'save_dir', 'save'),
        ('star', 'save', 'save'),
    ]

    # Apply these overrides
    for section, key, value in optional_overrides:
        # To avoid too much noise in headers, only
        # copy over sections for samplers we're actually
        # using
        if section not in sampler_names:
            continue
        # Only override the option if it is already set
        if (section, key) in cosmosis_options:
            full_value = output_dir / value
            # Still warn the user
            sys.stderr.write(f"NOTE: Overriding option {section}/{key}"
                             f" to {full_value}")
            cosmosis_options[section, key] = str(full_value)

    # The string parameters in the yaml file parameters
    # can't go into cosmosis values, because that is for parameters
    # that might vary during a run, which string params will not.
    # Instead we put these in the parameter file
    for p, v in config['parameters'].items():
        if isinstance(v, str):
            cosmosis_options['firecrown', p] = v

    # Convert into cosmosis Inifile format.
    cosmosis_params = cosmosis.Inifile(None, override=cosmosis_options)

    return cosmosis_params


def _make_cosmosis_values(config):
    """Extract a cosmosis values object from a config dict

    Parameters
    ----------
    config : dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function

    Returns
    -------
    cosmosis_values : Inifile
        Object to use to build cosmosis parameter ranges/values.
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
    """Make a cosmosis priors ini file.

    Parameters
    ----------
    config : dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function

    Returns
    -------
    priors : cosmosis Inifile
        The cosmosis config object specifying priors.
    """

    # Early return if no priors section is specified
    if 'priors' not in config:
        return cosmosis.Inifile(None)

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


def _setup(data_ini):
    # Most CosmoSIS modules do proper setup here.
    # We don't need amything so just return.
    return data_ini


def _execute(block, config):
    data, ini = config
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

    # Currently compute_loglike actually computes the posterior
    # if priors are included. Prevent that from happening since
    # CosmoSIS is already handling priors
    if 'priors' in data:
        data = data.copy()
        del data['priors']

    # Call out to the log likelihood
    loglikes, obs, theory, covs, invs, stats = compute_loglike(cosmo=cosmo, data=data)
    loglike = np.sum([v for v in loglikes.values() if v is not None])

    # For Fisher, etc., we save all the data vector info that we have
    for name in loglikes:
        # skip some stuff
        if name in RESERVED_NAMES_COSMOSIS:
            continue

        # Send result back to cosmosis
        block['likelihoods', f'{name}_like'] = loglikes[name]

        # Save whatever we have managed to collect.
        # The CosmoSIS Fisher sampler and others look in this
        # section to build up the Fisher data vectors.
        if theory[name] is not None:
            block['data_vector', f'{name}_theory'] = theory[name]
        if obs[name] is not None:
            block['data_vector', f'{name}_data'] = obs[name]
        if covs[name] is not None:
            block['data_vector', f'{name}_covariance'] = covs[name]
        if invs[name] is not None:
            block['data_vector', f'{name}_inverse_covariance'] = invs[name]

    # Unless in quiet mode, print out what we have done
    if not data['cosmosis'].get("quiet", True):
        print("params = {}".format(data['parameters']))
        print(f"loglike = {loglike}\n", flush=True)

    # Signal success.  An exception anywhere above will
    # be converted to a -inf likelihood by default.
    return 0
