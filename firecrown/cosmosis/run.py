import os
import sys
import numbers
import warnings
from ..cosmology import get_ccl_cosmology, RESERVED_CCL_PARAMS
from ..loglike import compute_loglike
import numpy as np

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

    Will look for the 'mpi' key in the config cosmosis section.

    Parameters
    ----------
    config: dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function

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

    ini: Inifile
        Cosmosis object representing the main input parameter file

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


def _make_cosmosis_params(config):
    """Extract a cosmosis configuration object from a config dict

    Parameters
    ----------
    config: dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function

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

    # Convert into cosmosis Inifile format.
    cosmosis_params = cosmosis.Inifile(None, override=cosmosis_options)

    return cosmosis_params


def _make_cosmosis_values(config):
    """Extract a cosmosis values object from a config dict

    Parameters
    ----------
    config: dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function

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
    """Make a cosmosis priors ini file.

    Parameters
    ----------
    config: dict
        The data object parse'd from an input yaml file.
        This is passed as-is to the likelihood function

    Returns
    -------
    priors: cosmosis Inifile
        The cosmosis config object specifying priors
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


def _setup(data):
    # Most CosmoSIS modules do proper setup here.
    # In this module we just collect together the
    # covariances and get their inverses, so that
    # we can do a Fisher matrix later, if we want to.
    from cosmosis.runtime.utils import symmetric_positive_definite_inverse
    data, ini = data
    invs = {}
    covs = {}
    error = False
    for name, config in data.items():
        # ignore priors and other non-likelihood sections
        if name == 'priors' or 'data' not in config:
            continue

        # deal with any of
        # - there being no likelihood specified
        # - the likelihood not being a gaussian
        # If there is a better way of introspecting
        # this that would be great.
        try:
            cov = config['data']['likelihood'].cov
        except (AttributeError, KeyError):
            error = True
            continue

        # Get inverse if possible. Might not be SPD,
        # though it should be.  We allow this for most samplers
        # because small errors can creep in numerically, but we
        # don't allow for Fisher
        if cov is None:
            inv_cov = None
        else:
            try:
                inv_cov = symmetric_positive_definite_inverse(cov)
            except ValueError:
                error = True
                continue
        # If the above didn't work then we should already have
        # continue'd, so if we get this far all is good.
        covs[name] = cov
        invs[name] = inv_cov

    if error:
        warnings.warn("Note that not all of your likelihoods are "
                      "valid Gaussians, so I will not be able to "
                      "run Fisher matrix, if that's what you wanted.")

    return data, ini, covs, invs


def _execute(block, config):
    data, ini, covs, invs = config
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
    loglike, stats = compute_loglike(cosmo=cosmo, data=data)

    # concatenate theory and data vectors, where these
    # are supported by the log likelihood
    theory = {}
    obs = {}
    for name, stat in stats.items():
        # These can easily be missing, in which case they will just
        # be left out.
        try:
            obs[name] = np.concatenate([v for v in stat['data'].values()])
            theory[name] = np.concatenate([v for v in stat['theory'].values()])
        except (KeyError, ValueError):
            pass

    # For Fisher, etc., we save all the data vector info that we have
    for name in data:
        # indicates that this is a likelihood
        if 'data' not in data[name]:
            continue

        # Send result back to cosmosis
        block['likelihoods', f'{name}_like'] = stats[name]['loglike']

        # Save whatever we have managed to collect.
        # The CosmoSIS Fisher sampler and others look in this
        # section to build up the Fisher data vectors.
        if name in theory:
            block['data_vector', f'{name}_theory'] = theory[name]
        if name in obs:
            block['data_vector', f'{name}_data'] = obs[name]
        if name in covs:
            block['data_vector', f'{name}_covariance'] = covs[name]
        if name in invs:
            block['data_vector', f'{name}_inverse_covariance'] = invs[name]

    # Unless in quiet mode, print out what we have done
    if not data['cosmosis'].get("quiet", True):
        print("params = {}".format(data['parameters']))
        print(f"loglike = {loglike}\n", flush=True)

    # Signal success.  An exception anywhere above will
    # be converted to a -inf likelihood by default.
    return 0
