"""Power spectrum calculation functions for two-point theory."""

import numpy as np
import pyccl

from firecrown.likelihood.source import Tracer
from firecrown.modeling_tools import ModelingTools


def calculate_pk(
    pk_name: str, tools: ModelingTools, tracer0: Tracer, tracer1: Tracer
) -> pyccl.Pk2D:
    """Return the power spectrum named by pk_name.

    If the modeling tools already has the power spectrum, it is returned.
    If not, is is computed with the help of the modeling tools.

    :param pk_name: The name of the power spectrum to return.
    :param tools: The modeling tools to use.
    :param tracer0: The first tracer to use.
    :param tracer1: The second tracer to use.
    :return: The power spectrum.
    """
    if tools.has_pk(pk_name):
        # Use existing power spectrum
        pk = tools.get_pk(pk_name)
    elif tracer0.has_pt or tracer1.has_pt:
        pk = at_least_one_tracer_has_pt(tools, tracer0, tracer1)
    elif tracer0.has_hm or tracer1.has_hm:
        pk = at_least_one_tracer_has_hm(tools, tracer0, tracer1)
    else:
        raise ValueError(f"No power spectrum for {pk_name} can be found.")
    return pk


def at_least_one_tracer_has_hm(
    tools: ModelingTools, tracer0: Tracer, tracer1: Tracer
) -> pyccl.Pk2D:
    """Compute a power spectrum with the halo model.

    :param tools: The modeling tools to use.
    :param tracer0: The first tracer to use.
    :param tracer1: The second tracer to use.
    :return: The power spectrum.
    """
    # Compute halo model power spectrum
    # Fix a_arr because normalization is zero for a<~0.07
    # TODO: Test if a_arr sampling is enough.
    a_arr = np.linspace(0.1, 1, 16)
    ccl_cosmo = tools.get_ccl_cosmology()
    hm_calculator = tools.get_hm_calculator()
    cM_relation = tools.get_cM_relation()
    IA_bias_exponent = (
        2  # Square IA bias if both tracers are HM (doing II correlation).
    )
    if not (tracer0.has_hm and tracer1.has_hm):
        assert "shear" in [tracer0.tracer_name, tracer1.tracer_name], (
            "Currently, only cosmic shear is supported "
            "with the halo model for intrinsic alignments."
        )
        IA_bias_exponent = (
            1  # IA bias if not both tracers are HM (doing GI correlation).
        )
        # mypy complains about the following line even though
        # the HMCalculator type does have a mass_def attribute.
        other_profile = pyccl.halos.HaloProfileNFW(
            mass_def=hm_calculator.mass_def,
            concentration=cM_relation,
            truncated=True,
            fourier_analytic=True,
        )
        other_profile.ia_a_2h = -1.0  # used in GI contribution, which is negative.
        if not tracer0.has_hm:
            assert tracer1.halo_profile is not None
            profile0: pyccl.halos.HaloProfile = other_profile
            profile1: pyccl.halos.HaloProfile = tracer1.halo_profile
        else:
            assert tracer0.halo_profile is not None
            profile0 = tracer0.halo_profile
            profile1 = other_profile
    else:
        assert tracer0.halo_profile is not None
        assert tracer1.halo_profile is not None
        profile0 = tracer0.halo_profile
        profile1 = tracer1.halo_profile
    # Ensure that profile0 and profile1 are not None.
    assert profile0 is not None
    assert profile1 is not None
    # Compute here the 1-halo power spectrum
    pk_1h = pyccl.halos.halomod_Pk2D(
        cosmo=ccl_cosmo,
        hmc=hm_calculator,
        prof=profile0,
        prof2=profile1,
        a_arr=a_arr,
        get_2h=False,
    )
    # Compute here the 2-halo power spectrum
    C1rhocrit = (
        5e-14 * pyccl.physical_constants.RHO_CRITICAL
    )  # standard IA normalization
    # These assertions are required because the pyccl profiles do not have ia_a_2h.
    # That is something added locally.
    assert hasattr(profile0, "ia_a_2h")
    assert hasattr(profile1, "ia_a_2h")
    assert hasattr(ccl_cosmo, "growth_factor")
    assert hasattr(ccl_cosmo, "nonlin_matter_power")
    pk_2h = pyccl.Pk2D.from_function(
        pkfunc=lambda k, a: profile0.ia_a_2h
        * profile1.ia_a_2h
        * (C1rhocrit * ccl_cosmo["Omega_m"] / ccl_cosmo.growth_factor(a))
        ** IA_bias_exponent
        * ccl_cosmo.nonlin_matter_power(k, a),
        is_logp=False,
    )
    pk = pk_1h + pk_2h
    return pk


def at_least_one_tracer_has_pt(
    tools: ModelingTools, tracer0: Tracer, tracer1: Tracer
) -> pyccl.Pk2D:
    """Compute the power spectrum with the perturbation theory.

    If one of the tracers does not have a perturbation theory (PT) tracer, a dummy
    matter PT tracer is created for it. This is useful for doing cross-correlations
    between a PT tracer and a non-PT tracer.

    :param tools: The modeling tools to use.
    :param tracer0: The first tracer to use.
    :param tracer1: The second tracer to use.
    :return: The power spectrum.
    """
    if not (tracer0.has_pt and tracer1.has_pt):
        # Mixture of PT and non-PT tracers
        # Create a dummy matter PT tracer for the non-PT part
        matter_pt_tracer = pyccl.nl_pt.PTMatterTracer()
        if not tracer0.has_pt:
            tracer0.pt_tracer = matter_pt_tracer
        else:
            tracer1.pt_tracer = matter_pt_tracer
    # Compute perturbation power spectrum
    pt_calculator = tools.get_pt_calculator()
    pk = pt_calculator.get_biased_pk2d(
        tracer1=tracer0.pt_tracer,
        tracer2=tracer1.pt_tracer,
    )
    return pk
