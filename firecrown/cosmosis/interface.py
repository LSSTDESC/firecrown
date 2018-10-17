# For technical reasons this does have to be an absolute
# import. This has no effect on anything in this case 
# - the module is not reloaded
import firecrown
import pyccl

def setup(data):
    return data

def execute(block, data):
    # TODO: Replace this with proper parameter handling when ready
    required_params = ['Omega_k', 'Omega_b', 'Omega_c', 'h', 'n_s', 'A_s', 'w0', 'wa']
    params = {p: block['params', p] for p in required_params}
    cosmo = pyccl.Cosmology(**params)

    # Call out to the log likelihood
    loglike, stats = firecrown.compute_loglike(cosmo=cosmo, data=data)
    
    # Send result back to cosmosis
    block['likelihoods', 'firecrown_like'] = loglike

    # Unless in quiet mode, print out what we have done
    if not data['sampler']['quiet']:
        print(f"params = {params}\nloglike = {loglike}\n")

    # Signal success.  An exception anywhere above will
    # be converted to an error status
    return 0
