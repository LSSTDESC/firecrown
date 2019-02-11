import functools
import pyccl

RESERVED_CCL_PARAMS = (
    "Omega_c",
    "Omega_b",
    "h",
    "A_s",
    "sigma8",
    "n_s",
    "Omega_k",
    "Omega_g",
    "w0",
    "wa",
    "bcm_log10Mc",
    "bcm_etab",
    "bcm_ks",
    "transfer_function",
    "matter_power_spectrum",
    "baryons_power_spectrum",
    "mass_function",
    "halo_concentration",
    "emulator_neutrinos")

# FIXME: these params are not supported right now
# Neff (:obj:`float`, optional): Effective number of massless
#     neutrinos present. Defaults to 3.046.
# m_nu (:obj:`float`, optional): Total mass in eV of the massive
#     neutrinos present. Defaults to 0.
# mnu_type (:obj:`str`, optional): The type of massive neutrinos.
# df_mg (array_like, optional): Perturbations to the GR growth rate as
#     a function of redshift :math:`\\Delta f`. Used to implement simple
#     modified growth scenarios.
# z_mg (array_like, optional): Array of redshifts corresponding to df_mg.


@functools.lru_cache(maxsize=1024)
def _get_ccl_cosmology(params):
    dct = {p: v for p, v in params}
    return pyccl.Cosmology(**dct)


def get_ccl_cosmology(input_params):
    """Get an input CCL cosmology using a cache.

    Parameters
    ----------
    input_params : dict
        An input parameter dictionary. The name of the parameters should
        match the CCL name. Parameters not in the CCL namespace are ignored.
        The value can be a float or a list of three floats. In the case that
        a list is passed, the middle value is used.

    Returns
    -------
    cosmo : pyccl.Cosmology
        The CCL Cosmology object.
    """
    params = []
    for p, val in input_params.items():
        if p in RESERVED_CCL_PARAMS:
            if isinstance(p, list) and not isinstance(p, str):
                params.append((p, val[1]))
            else:
                params.append((p, val))
    # we are using a tuple here since this type is hashable and so can be
    # used a key for lru_cache
    return _get_ccl_cosmology(tuple(params))
