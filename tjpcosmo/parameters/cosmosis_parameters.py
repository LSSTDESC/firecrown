from cosmosis.datablock import names
from .parameter_set import ParameterSet

# Translate Cosmosis blocks to PHIL PARAMS!!!
def block_to_parameters(block, consistency):
    """ This fucntion translates the parameters from a cosmosis block to TJPCosmo's
    Parameter class (A sub class of a dictionary. This list should be consistent 
    with the DESC standard for cosmological parameters. A few parameters are 
    mandatory while others will be getting defaults values if not specified in 
    the initial file.
    """
    Omega_c = block[names.cosmological_parameters, 'Omega_c']
    Omega_b = block[names.cosmological_parameters, 'Omega_b']
    h = block[names.cosmological_parameters, 'h']
    n_s = block[names.cosmological_parameters, 'n_s']


    
    #Optional parameters, will be set to a default value, if not there
    A_s = block.get_double(names.cosmological_parameters, 'a_s', 0.0)
    sigma_8 = block.get_double(names.cosmological_parameters, 'sigma_8', 0.0)
    
    
    w0 = block.get_double(names.cosmological_parameters, 'w0',-1.0)
    wa = block.get_double(names.cosmological_parameters, 'wa', 0.0)
    
    Omega_n_mass = block.get_double(names.cosmological_parameters, 'Omega_n_mass', 0.0)
    Omega_n_rel = block.get_double(names.cosmological_parameters, 'Omega_n_rel', 0.0)
    Omega_g = block.get_double(names.cosmological_parameters, 'Omega_g', 0.0)
    N_nu_mass = block.get_double(names.cosmological_parameters, 'N_nu_mass', 0.0)
    N_nu_rel = block.get_double(names.cosmological_parameters, 'N_nu_rel', 3.046)
    mnu = block.get_double(names.cosmological_parameters, 'mnu', 0.0)
    
    # Now if we have provided the code with Omega_k or Omega_l it will figure out
    known_parameters = {}
    for param in consistency.parameters:
        if block.has_value("cosmological_parameters", param):
            known_parameters[param] = block["cosmological_parameters", param]


    cosmo_parameters = consistency(known_parameters)
    #parameters = ParameterSet(**cosmo_parameters)

    cosmo_parameters['w0'] = w0
    cosmo_parameters['wa'] = wa

    cosmo_parameters['omega_n_rel'] = Omega_n_rel
    cosmo_parameters['omega_g'] = Omega_g
    cosmo_parameters['mnu'] = mnu
    cosmo_parameters['n_nu_mass'] = N_nu_mass
    cosmo_parameters['n_nu_rel'] = N_nu_rel
    cosmo_parameters['n_s'] = n_s

    if A_s!=0:
        cosmo_parameters['a_s'] = A_s
    if sigma_8!=0:
        cosmo_parameters['sigma_8'] = sigma_8

    
    


    sections = {}
    for section in block.sections():
        if section=="cosmological_parameters":
            continue
        p = {}
        keys = block.keys(section)
        for _,key in keys:
            p[key] = block[section,key]
        sections[section] = ParameterSet(**p)


    # Omega_l = full_parameters["omega_lambda"]
    parameters = ParameterSet(**cosmo_parameters, **sections)

    return parameters    
    #Everything done so far gets thrown into the to DESC standard cosmoogy base.
    # cosmology = CosmoBase(Omega_c, Omega_b, Omega_l, h, n_s, A_s, sigma_8, Omega_g,
    #     Omega_n_mass, Omega_n_rel, w0, wa, N_nu_mass, N_nu_rel, mnu)
    
    # return cosmology

