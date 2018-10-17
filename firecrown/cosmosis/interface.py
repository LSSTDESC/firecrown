
import pyccl
import firecrown

def setup(data):
    print(data)
    return data

def execute(block, data):
    # Replace with proper parameter handling when ready
    required_params = ['Omega_k', 'Omega_b', 'Omega_c', 'h', 'n_s', 'A_s', 'w0', 'wa']
    params = {p: block['params', p] for p in required_params}
    print(params)
    cosmo = pyccl.Cosmology(**params)
    print(cosmo)
    loglike, stats = firecrown.compute_loglike(cosmo=cosmo, data=data)
    block['likelihoods', 'firecrown_like'] = loglike

    return 0
