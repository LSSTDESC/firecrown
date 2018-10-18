# For technical reasons this does have to be an absolute
# import. This has no effect on anything in this case
# - the module is not reloaded
import firecrown
import pyccl
import sys


def setup(data):
    return data


def execute(block, data):

    # Create CCL cosmology
    ccl_params = ['Omega_k', 'Omega_b', 'Omega_c',
                  'h', 'n_s', 'A_s', 'w0', 'wa']
    ccl_values = {p: block['params', p] for p in ccl_params}
    cosmo = pyccl.Cosmology(**ccl_values)

    all_params = data['parameters'].keys()
    for p in all_params:
        data['parameters'][p] = block['params', p]

    # Call out to the log likelihood
    loglike, stats = firecrown.compute_loglike(cosmo=cosmo, data=data)

    # Send result back to cosmosis
    block['likelihoods', 'firecrown_like'] = loglike

    # Unless in quiet mode, print out what we have done
    if not data['sampler']['quiet']:
        print("params = {}".format(data['parameters']))
        print(f"loglike = {loglike}\n")
        # Useful to flush when running under MPI, to avoid
        # buffering
        sys.stdout.flush()

    # Signal success.  An exception anywhere above will
    # be converted to a -inf likelihood by default.
    return 0
