import argparse
import os
import pathlib
import pdb

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
parser.add_argument("inifile", help="Input ini file of parameters")
parser.add_argument("--mpi",action='store_true',help="Run in MPI mode.")
parser.add_argument("--smp",type=int,default=0,help="Run with the given number of processes in shared memory multiprocessing (this is experimental and does not work for multinest).")
parser.add_argument("--pdb",action='store_true',help="Start the python debugger on an uncaught error. Only in serial mode.")


def list_samplers():
    print("Samplers:")
    for sampler in cosmosis.samplers.sampler.RegisteredSampler.registry:
        print("    " + sampler)
    return 0



def main(args):

    args = parser.parse_args(args)

    # A handy option to list all available samplers
    if args.list_samplers:
        return list_samplers()


    # Set a number of default parameters that cosmosis wants that are 
    # always the same for TJCosmo
    override = {
        # This parameter just needs a dummy value to stop cosmosis complaining
        ("runtime","root") : "",
        # Our models are always made up of a theory component and a likelihood components
        ("pipeline","modules") : "model",
        # we always generate one named (total) likelihood, called "lsst"
        ("pipeline","likelihoods") : "total",
        # we always use these same two cosmosis modules, but we configure them 
        # to do what we want.
        ("model","file") : "model",
        ("model","file") : str(dirname.joinpath('theory_model.py')),
    }
    args.params = override
    args.variables = {}

    # Run the cosmosis main function, optionally under two different types of parallelism
    if args.mpi:
        with mpi_pool.MPIPool() as pool:
            run_cosmosis(args,pool)
    elif args.smp:
        with process_pool.Pool(args.smp) as pool:
            run_cosmosis(args,pool)
    # or just in serial, with the error handling
    else:
        try:
            run_cosmosis(args)
        except Exception as error:
            if args.pdb:
                print("pdb")
                print("There was an exception - starting python debugger because you ran with --pdb")
                print(error)
                pdb.post_mortem()
            else:
                print("here")
                raise
