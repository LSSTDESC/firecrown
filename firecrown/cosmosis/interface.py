# For technical reasons this does have to be an absolute
# import. This has no effect on anything in this case
# - the module is not reloaded
import firecrown
from firecrown.cosmology import get_ccl_cosmology, RESERVED_CCL_PARAMS


def setup(data):
    # In most cosmosis modules this function
    # is a bit more complicated!  We are hacking it
    # a bit here - normally the input to this function
    # is the configuration information for the module.
    return data


def execute(block, data):
    # Calculate the firecrown likelihood as a module
    # This function, which isn't designed for end users,
    # is the main connection between cosmosis and firecrown.
    # CosmoSIS builds the block, and passes it to us here.
    # The block contains all the sample parameters.

    # Create CCL cosmology
    ccl_values = {}
    for p in RESERVED_CCL_PARAMS:
        try:
            val = block['params', p]
            ccl_values[p] = val
        except Exception:
            pass
    cosmo = get_ccl_cosmology(ccl_values)

    # Put all the parameters in the data dictionary,
    # both CCL-related and others, like nuisance params.
    all_params = data['parameters'].keys()
    for p in all_params:
        data['parameters'][p] = block['params', p]

    # Call out to the log likelihood
    loglike, stats = firecrown.compute_loglike(cosmo=cosmo, data=data)

    # Send result back to cosmosis
    block['likelihoods', 'firecrown_like'] = loglike

    # Unless in quiet mode, print out what we have done
    if not data['cosmosis'].get("quiet", True):
        print("params = {}".format(data['parameters']))
        print(f"loglike = {loglike}\n", flush=True)

    # Signal success.  An exception anywhere above will
    # be converted to a -inf likelihood by default.
    return 0
