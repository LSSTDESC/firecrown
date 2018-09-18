"""
These are very thin cosmosis wrappers that connect to tell it how to connect
to the primary TJPCosmo code.

"""
from cosmosis.datablock import names, option_section

# We do need to do absolute imports here, for technical reasons to do with
# how cosmosis loads modules.  That also means that no other TJPCosmo module
# should ever import this one.
from tjpcosmo.analyses import Analysis, convert_cosmobase_to_ccl
from tjpcosmo.parameters import ParameterSet, CosmoBase, cosmology_consistency
from tjpcosmo.parameters.cosmosis_parameters import block_to_parameters
from tjpcosmo.likelihood import Gaussian
import pathlib
import yaml
import numpy as np
import pyccl as ccl


def setup(options):
    """
    Sets up the input to cosmosis for each analysis model.

    This function is called once at the start of the run.
    """

    # Find the YAML file that describes our analyses.
    config_filename = options.get_string(option_section, "config")
    save_data_to_cosmosis = options.get_bool(option_section, "save_data_to_cosmosis")
    path = pathlib.Path(config_filename).expanduser()
    config = yaml.load(path.open())
    
    # The main workhorse of this code is the Analysis objects,
    # which connect together likelihoods, data, theory calculations,
    # etc.
    analyses = [
        Analysis.from_dict(name, config[name]) 
        for name in config['correlated_probes']]

    # The other thing we create and pass along is a consistency-enforcer,
    # which 
    consistency = cosmology_consistency()

    # These two objects will be give to the "execute" function 
    # below when it is called later.
    return analyses, consistency, save_data_to_cosmosis

def execute(block, config):
    """
    Run all the analyses in the 

    This function is called once per parameter set
    """
    analyses, consistency, save_data_to_cosmosis = config
    parameterSet = block_to_parameters(block, consistency)
    params = convert_cosmobase_to_ccl(parameterSet)

    print("Calling CCL with default config - may need to change depending on systematics/choices")
    cosmo=ccl.Cosmology(params)

    total_like = 0.0
    for analysis in analyses:
        like, theory_result = analysis.run(cosmo, parameterSet)

        if save_data_to_cosmosis:
            theory_result.to_cosmosis_block(block, analysis.name)
            if isinstance(analysis.likelihood, Gaussian):
                P = analysis.likelihood.data.precision
                block['data_vector', analysis.name+'_inverse_covariance'] = P

        # We always need the likelihood
        block['likelihoods', analysis.name+'_like'] = like

        # Collect the total likelihood value
    return 0

