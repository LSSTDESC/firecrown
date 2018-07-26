import argparse
import os
import pathlib
import pdb
import sys
import yaml
# external packages
from cosmosis.main import run_cosmosis, mpi_pool, process_pool, Inifile
import cosmosis.samplers

# internal imports
from . import utils



dirname = pathlib.Path(__file__).parent



description = """
The TJPCosmo Cosmology constraints code.

TJPCosmo uses The Core Cosmology Library (CCL) for its cosmology predictions
and CosmoSIS for its sampling, infrastructure, and user interface.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('--list-samplers', action='store_true', help='List all available sampling methods.')
parser.add_argument("yaml_config_file", help="Input YAML configuration file describing run")
parser.add_argument("--mpi",action='store_true',help="Run in MPI mode.")
parser.add_argument("--smp",type=int,default=0,help="Run with the given number of processes in shared memory multiprocessing (this is experimental and does not work for multinest).")
parser.add_argument("--pdb",action='store_true',help="Start the python debugger on an uncaught error. Only in serial mode.")


def list_samplers():
    print("Samplers:")
    for sampler in cosmosis.samplers.sampler.RegisteredSampler.registry:
        print("    " + sampler)
    return 0



def main(args):
    """ Main run loop, calls cosmosis to do the sampling. 
    Reads in the command line argument, which should be the .ini file used for this run.
    Look at example for that Params.ini for how one of these are structured.
    
    Cosmosis calls the theory_model.py for information on how to execute the TJPCosmo code.
    """
    args = parser.parse_args(args)

    # A handy option to list all available samplers
    if args.list_samplers:
        return list_samplers()

    # Load the configuration file.
    # We actually end up opening this twice at the moment, once here
    # to configure cosmosis and its samplers, and once in the cosmosis_entry_point file
    # to configure the module and calculation
    path = pathlib.Path(args.yaml_config_file).expanduser()
    config = yaml.load(path.open())

    sampling = config['sampling']

    sampler = sampling['sampler']

    # Most samplers only need the likelihoods, but the Fisher and
    # test samplers benefit from getting all the data
    save_data_to_cosmosis = sampler in ['fisher', 'test']


    override = {
        # This parameter just needs a dummy value to stop cosmosis complaining
        ("runtime","root") : "",
        # Some parameters have slightly more intuitive names here
        ("runtime","sampler") : sampler,
        # we always generate one named (total) likelihood
        # In the newest cosmosis all detected likelihoods are found
        # ("pipeline","likelihoods") : "total",
        # For now always be noisy
        ("pipeline","quiet") : "F",
        # we always use regard our entire pipeline as a single cosmosis module
        # but we can configure it elsewhere.  The module name is just "model"
        ("pipeline","modules") : "model",
        #
        ("model","file") : str(dirname.joinpath('cosmosis_entry_point.py')),
        ("model","config") : args.yaml_config_file,
        ("model","save_data_to_cosmosis") : str(save_data_to_cosmosis)[0],
    }

    # Take the parts of the config file that define the sampling and put them
    # in cosmosis format
    for section, params in sampling.items():
        if section=='sampler':
            continue
        for key, value in params.items():
            override[(section,key)] = str(value)

    for key, value in config['output'].items():
        override[('output', key)] = str(value)


    ini = Inifile(None, override=override)


    cosmosis_args = argparse.Namespace()
    cosmosis_args.inifile = ""
    cosmosis_args.params = {}
    cosmosis_args.variables = {}
    cosmosis_args.experimental_fault_handling = False


    values_override = {}
    for section, params in config['parameters'].items():
        for key, value in params.items():
            if isinstance(value, list):
                value = "  ".join(str(s) for s in value)
            else:
                value = str(value)
            values_override[(section,key)] = value

    values = Inifile(None, override=values_override)



    # Run the cosmosis main function, optionally under two different types of parallelism
    if args.mpi:
        with mpi_pool.MPIPool() as pool:
            run_cosmosis(cosmosis_args,pool,ini=ini,values=values)
    elif args.smp:
        with process_pool.Pool(args.smp) as pool:
            run_cosmosis(cosmosis_args,pool,ini=ini,values=values)
    # or just in serial, with the error handling
    else:
        try:
            run_cosmosis(cosmosis_args,ini=ini,values=values)
        except Exception as error:
            if args.pdb:
                print("pdb")
                print("There was an exception - starting python debugger because you ran with --pdb")
                print(error)
                pdb.post_mortem()
            else:
                print("here")
                raise
