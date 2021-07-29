from cosmosis.datablock import option_section
from cosmosis.datablock import names as section_names
import numpy as np
import pyccl as ccl
import firecrown
from firecrown.convert import firecrown_convert_builder
from firecrown.connector.mapping import from_cosmosis_camb, redshift_to_scale_factor

likes = section_names.likelihoods


def calculate_background(sample):
    """Calculate the background (dictionary) required for CCL, from a CosmoSIS
    datablock."""
    a = np.flip(1.0 / (1.0 + sample["distances", "z"]))
    chi = np.flip(sample["distances", "d_m"])
    # TODO: is this scaling correct?
    h0 = sample["cosmological_parameters", "h0"]
    speed_of_light = 2
    h_over_h0 = np.flip(sample["distances", "h"]) * ccl.physical_constants.CLIGHT / h0
    return {"a": a, "chi": chi, "h_over_h0": h_over_h0}


class FirecrownLikelihood:
    """CosmoSIS likelihood module for calculating Firecrown likelihood.

    In this simplest implementation, we have only a single module. This module
    is responsible for calling CCL to perform theory calculations, based on the
    output of CAMB, and also for calculating the data likelihood baesd on this
    theory.
    """

    def __init__(self, config):
        # Capture the result of firecrown.
        firecrown_yaml_file = config[option_section, "firecrown_config"]
        _, self.data = firecrown.parse(firecrown_yaml_file)
        assert type(self.data) is dict

        # TODO: CCLPrecisionParameters object instead of this glue code.
        # Consider migrating this to CCL itself.

        # Hardwire to use CAMB module. We should get this from the datablock:
        # look for either CAMB or CLASS, and fail if one or the other is not
        # specified.
        self.translation_mode = "CAMB"

        # This specification of the knots needs to go into the CAMB
        # configuration.œ Does CLASS have the same capability?
        #
        # The following is what is done for Cobaya. We need to get from CAMB
        # what the configuration was, and remember them for our owm use when we
        # call CCL. If we can just *get the knots* from CAMB itself, we should
        # use them.

        self.fc_params = firecrown_convert_builder(input_style=self.translation_mode)

    def __str__(self):
        """Return the human-readabe representation of this object."""
        return f"Firecrown object with keys: {list(self.data.keys())}"

    def execute(self, sample):
        # We have to make a new ccl object on each sample.
        # Get CAMB output; look at ccl.get_requirements to see what is required.

        # Get all cosmological parameters from the sample.
        cosmological_parameter_names = [
            name
            for section, name in sample.keys()
            if section == "cosmological_parameters"
        ]
        cosmological_params = {
            name: sample["cosmological_parameters", name]
            for name in cosmological_parameter_names
        }


        cosmological_params_for_ccl = from_cosmosis_camb(cosmological_params)


        h0 = cosmological_params["h0"]
        k = sample["matter_power_lin", "k_h"] * h0
        z = sample["matter_power_lin", "z"]
        p_k = sample["matter_power_lin", "p_k"] / (h0 ** 3)

        scale, p_k = redshift_to_scale_factor(z, p_k)

        # TODO: also handle the non-linear; we need configurability
        # We need 3 configuration modes:
        #     CAMB linear only
        #     CAMB linear + CAMB nonlinear
        #     CAMB linear + CCL nonlinear

        background = calculate_background(sample)

        cosmo = ccl.CosmologyCalculator(
            **cosmological_params_for_ccl.asdict(),
            background=background,
            pk_linear={"a": scale, "k": k, "delta_matter:delta_matter": p_k},
            # TODO: the nonlinear_model should be a configuration parameter; this
            # is part of the configuration above.
            # TODO: if we are using CCL to calculate the nonlinear power spectrum,
            # we probably should not have CAMB configured to give us a nonlinear power
            # spectrum. Emit a warning? Fail?
            nonlinear_model="halofit",
        )

        # TODO:
        #   1. figure out if we were using CAMB or CLASS to do Boltzmann calculations
        #   2. convert cosmological_params (as they came from CosmoSIS, which either means CAMB or CLASS ... right?)
        #      to CCL format, to be fed to firecrown.compute_loglike. Create a CCLCosmologicalParams class or dataclass for this.
        #      This class should be constructible from the same set of parameters as the pyccl.core.Cosmology (or xxCalculator?)
        #      class. All keyword args for __init__? That would allow initialzation from a splatted dictionary. Add a factory
        #      function from a pyccl.core.Cosmology object itself?
        #   3. Get the Boltzmann calculations from the datablock; convert from CAMB format to CCL calculator mode versions. Need
        #      to get units right, need to reverse the order of entries in power spectrum. NO need to make things calculated by
        #      CCL visible to CosmoSIS at this stage (except for the likelihood).
        #   4. Call firecrown.compute_loglike
        #   5. put the resulting likelihood into the datablock

        lnlikes, *_ = firecrown.compute_loglike(cosmo=cosmo, data=self.data)
        lnlike = np.sum(v for v in lnlikes.values() if v is not None)

        sample.put_double(section_names.likelihoods, "firecrown_like", lnlike)
        return 0

    def cleanup(self):
        """There is nothing to do in the cleanup function for this module."""
        return 0

    # def _make_ccl_cosmology(self, sample):
    #     """Call pyccl to create the cosmology for this sample."""
    #     # Get stuff from the sample for the current sample; output from CAMB
    #
    #     # a_bg = np.linspace(0.1, 1.0, 50)
    #     # z_bg = 1.0 / self.a_bg - 1.0
    #     # z_Pk = np.arange(0.2, 6.0, 1)
    #     # Pk_kmax = 1.0
    #
    #     # Translate CAMB output to CCL format
    #     # Create the CCL cosmology for this sample


def setup(config):
    return FirecrownLikelihood(config)


def execute(sample, instance):
    return instance.execute(sample)


def cleanup(instance):
    return instance.cleanup()
