from cosmosis.datablock import option_section
import numpy as np
import firecrown
from firecrown.convert import firecrown_convert_builder

import logging

logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
)


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
        logging.debug("FirecrownLikelihood created.")
        logging.debug(self)
        logging.debug(f"Parameters are: {self.data['parameters']}")

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

        cosmo = self._make_ccl_cosmology(sample)
        likelihood = firecrown.compute_loglike(cosmo=cosmo, data=self.data)
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
