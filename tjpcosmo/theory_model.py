"""
These are very thin cosmosis wrappers that connect to tell it how to connect
to the primary TJPCosmo code.

"""
from cosmosis.datablock import names, option_section
from tjpcosmo.models import BaseModel
from tjpcosmo.likelihood import BaseLikelihood
from tjpcosmo.parameters import Parameters
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
    model_class = BaseModel.from_name(model_name)
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



def block_to_parameters(block):
    omega_b = block[names.cosmological_parameters, 'omega_b']
    h = block[names.cosmological_parameters, 'h']
    return Parameters(omega_b=omega_b, h=h)