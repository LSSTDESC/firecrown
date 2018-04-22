"""
These are very thin cosmosis wrappers that connect to tell it how to connect
to the primary TJPCosmo code.

"""
from cosmosis.datablock import names, option_section
from tjpcosmo.models import Analysis
from tjpcosmo.likelihood import BaseLikelihood
from tjpcosmo.parameters import ParameterSet
from Philscosmobase import CosmoBase
import pathlib
import yaml

def parse_data_set_options(options):
    data_files = options.get_string(option_section, "data")    
    data_info = {}
    for data_file in data_files.split():
        tag, section = data_file.split(':')
        d = {}
        for _, key in options.keys(section):
            d[key] = options[section,key]
        data_info[tag] = d
    return data_info


def setup(options):
    config_filename = options.get_string(option_section, "config")
    likelihood_name = options.get_string(option_section, "Likelihood")
    data_info = parse_data_set_options(options)

    path = pathlib.Path(config_filename).expanduser()
    config = yaml.load(path.open())


    # Get any metadata
    model_name = config['name']
    model_class = Analysis.from_name(model_name)
    likelihood_class = BaseLikelihood.from_name(likelihood_name)

    # Create the model using the yaml config info
    model = model_class(config, data_info, likelihood_class)
    # Return model and likelihood
    return model

def execute(block, model):
    # Generate a DESC Parameters object from a cosmosis block
    params = block_to_parameters(block)
    likelihood, theory_results = model.run(params)
    theory_results.to_cosmosis_block(block)
    block['likelihoods', 'total_like'] = likelihood
    return 0


# Translate Cosmosis blocks to PHIL PARAMS!!! MISSING IMPORT of phil's 
def block_to_parameters(block):
    # These are the mandatory parameters for cosmology, if they aren't there,
    # the code crashes.
    Omega_c = block[names.cosmological_parameters, 'Omega_c']
    Omega_b = block[names.cosmological_parameters, 'Omega_b']
    h = block[names.cosmological_parameters, 'h']
    n_s = block[names.cosmological_parameters, 'n_s']
    A_s = block[names.cosmological_parameters, 'A_s']

    
    #Optional parameters, will be set to a default value, if not there
    w0 = block.get_double(names.cosmological_parameters, 'w0',-1.0)
    wa = block.get_double(names.cosmological_parameters, 'wa', 0.0)
    
    Omega_n_mass = block.get_double(names.cosmological_parameters, 'Omega_n_mass', 0.0)
    Omega_n_rel = block.get_double(names.cosmological_parameters, 'Omega_n_rel', 0.0)
    Omega_g = block.get_double(names.cosmological_parameters, 'Omega_g', 0.0)
    N_nu_mass = block.get_double(names.cosmological_parameters, 'N_nu_mass', 0.0)
    N_nu_rel = block.get_double(names.cosmological_parameters, 'N_nu_rel', 3.046)
    mnu = block.get_double(names.cosmological_parameters, 'mnu', 0.0)
    sigma_8 = block.get_double(names.cosmological_parameters, 'Sigma_8', 0.0)
    
    #Parameters that must be derived
    Omega_m = Omega_c + Omega_b + Omega_n_mass
    
    #Either of These WE need to in the future be able to check which one 
    Omega_l = block[names.cosmological_parameters, 'Omega_l']  # NEED TO CHANGE THIS!
    Omega_k = 1.0 -(Omega_m + Omega_l + Omega_g + Omega_n_rel)		#NEED TO CHANGE THIS!
    
    Cosmology = CosmoBase(Omega_c, Omega_b, Omega_l, n_s, A_s, sigma_8, Omega_g,
        Omega_n_mass, Omega_n_rel, w0, wa, N_nu_mass, N_nu_rel, mnu)
    
    return Cosmology

