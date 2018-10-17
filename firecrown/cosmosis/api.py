import pathlib

# Locate the path to this directory
# For fiddly reasons that make a great deal of sense in an entirely
# different context to this one, we pass cosmosis a full path
# to the interface file
THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()
COSMOSIS_INTERFACE = str(THIS_DIRECTORY.joinpath('interface.py'))


def run(config, data):
    from cosmosis.runtime.module import Module
    from cosmosis.runtime.pipeline import LikelihoodPipeline
    from cosmosis.main import run_cosmosis
    from cosmosis.runtime.mpi_pool import MPIPool


    # Set up a single cosmosis module, which will be the interface
    # file in the same directory as this one
    module = Module('firecrown', COSMOSIS_INTERFACE)
    module.setup_functions(data)
    modules = [module]

    # Extract the bits of the config file that
    # cosmosis wants
    ini = make_cosmosis_config(config)
    values = make_cosmosis_values(config)

    # Whether to use MPI
    use_mpi = config['sampler'].get('mpi', False)
    if use_mpi:
        pool = MPIPool()
    else:
        pool = None
        print("Running single core")

    # Build the pipeline that evaluates the likelihood
    pipeline = LikelihoodPipeline(load=False, values=values, priors=None)
    pipeline.modules = modules

    # Actually run the thing
    run_cosmosis(None, pool=pool, ini=ini, pipeline=pipeline, values=values)


    if use_mpi:
        pool.close()


def make_cosmosis_config(config):
    from cosmosis.runtime.config import Inifile

    # Some general options
    config = config['sampler']
    sampler_name = config['sampler']
    output_file = config['output']
    debug = config.get('debug', False)
    quiet = config.get('quiet', False)
    root = "" # Dummy value to stop cosmosis complaining

    # Make into a pair dictionary with the right cosmosis sections
    cosmosis_options = {
        ("runtime","root") : root,  
        ("runtime","sampler") : sampler_name,
        ("output","filename") : output_file,
        ("pipeline","debug") : str(debug),
        ("pipeline","quiet") : str(quiet),
    }

    # Set all the sampler configuration options
    sampler_config = config.get(sampler_name, {})
    for key,val in sampler_config.items():
        cosmosis_options[(sampler_name, key)] = str(val)

    # Convert into cosmosis Inifile format.
    cosmosis_config = Inifile(None, override=cosmosis_options)

    return cosmosis_config


def make_cosmosis_values(config):
    from cosmosis.runtime.config import Inifile

    params = config['parameters']
    values = {}
    for p,v in params.items():
        key = ('params',p)
        if isinstance(v,list):
            values[key] = ' '.join(str(x) for x in v)
        else:
            values[key] = str(v)

    cosmosis_values = Inifile(None, override=values)

    return cosmosis_values


