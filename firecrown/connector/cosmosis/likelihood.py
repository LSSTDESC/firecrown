"""CosmoSIS Likelihood Connector.

This module provides the class FirecrownLikelihood, and the hook functions
for this module to be a CosmoSIS likelihood module.

Note that the class FirecrownLikelihood does *not* inherit from firecrown's
likelihood abstract base class; it the implementation of a CosmoSIS module,
not a specific likelihood.
"""

import cosmosis.datablock
from cosmosis.datablock import option_section
from cosmosis.datablock import names as section_names
import pyccl as ccl

from firecrown.connector.mapping import mapping_builder, MappingCosmoSIS
from firecrown.likelihood.gauss_family.gauss_family import GaussFamily
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.likelihood import load_likelihood, Likelihood, NamedParameters
from firecrown.parameters import ParamsMap
from firecrown.updatable import MissingSamplerParameterError


def extract_section(sample: cosmosis.datablock, section: str) -> NamedParameters:
    """Extract all the parameters from the name datablock section into a dictionary."""
    if not sample.has_section(section):
        raise RuntimeError(f"Datablock section `{section}' does not exist.")
    sec_dict = {name: sample[section, name] for _, name in sample.keys(section=section)}
    return NamedParameters(sec_dict)


class FirecrownLikelihood:
    """CosmoSIS likelihood module for calculating Firecrown likelihood.

    In this simplest implementation, we have only a single module. This module
    is responsible for calling CCL to perform theory calculations, based on the
    output of CAMB, and also for calculating the data likelihood based on this
    theory.

    :param config: current CosmoSIS datablock
    """

    def __init__(self, config: cosmosis.datablock):
        """Create the FirecrownLikelihood object from the given configuration."""
        likelihood_source = config.get_string(option_section, "likelihood_source", "")
        if likelihood_source == "":
            likelihood_source = config[option_section, "firecrown_config"]

        require_nonlinear_pk = config.get_bool(
            option_section, "require_nonlinear_pk", False
        )

        build_parameters = extract_section(config, option_section)

        sections = config.get_string(option_section, "sampling_parameters_sections", "")
        sections = sections.split()

        self.firecrown_module_name = option_section
        self.sampling_sections = sections
        self.likelihood: Likelihood
        try:
            self.likelihood, self.tools = load_likelihood(
                likelihood_source, build_parameters
            )
        except KeyError as err:
            print("*" * 30)
            print(f"The Firecrown likelihood needs a required parameter: {err}")
            print("*" * 30)
            raise
        self.map: MappingCosmoSIS = mapping_builder(
            input_style="CosmoSIS", require_nonlinear_pk=require_nonlinear_pk
        )

        # If sampling_sections  is empty, but we have required parameters, then
        # we have a configuration problem, and ParamsMap can never be built
        # correctly.
        if len(self.sampling_sections) == 0:
            required_parameters = (
                self.likelihood.required_parameters() + self.tools.required_parameters()
            )
            if len(required_parameters) != 0:
                msg = (
                    f"The configured likelihood has required "
                    f"parameters, but CosmoSIS is not providing them.\n"
                    f"The required parameters are:\n"
                    f"{list(required_parameters.get_params_names())}\n"
                    f"You need to provide the names of the DataBlock "
                    f"sections where these parameters are to be found\n"
                    f"in the `sampling_parameters_sections` parameter in the "
                    f"likelihood configuration."
                )
                raise RuntimeError(msg)

    def execute(self, sample: cosmosis.datablock) -> int:
        """This is the method called for each sample generated by the sampler."""
        cosmological_params: NamedParameters = extract_section(
            sample, "cosmological_parameters"
        )
        self.map.set_params_from_cosmosis(cosmological_params)

        ccl_args = self.map.calculate_ccl_args(sample)

        ccl_cosmo = ccl.CosmologyCalculator(**self.map.asdict(), **ccl_args)

        # TODO: Future development will need to capture elements that get put into the
        # datablock. This probably will be in a different "physics module" and not in
        # the likelihood module. And it requires updates to Firecrown to split the
        # calculations. e.g., data_vector/firecrown_theory  data_vector/firecrown_data

        firecrown_params = self.calculate_firecrown_params(sample)
        try:
            self.likelihood.update(firecrown_params)
            self.tools.update(firecrown_params)
        except MissingSamplerParameterError as exc:
            msg = self.form_error_message(exc)
            raise RuntimeError(msg) from exc

        self.tools.prepare(ccl_cosmo)
        loglike = self.likelihood.compute_loglike(self.tools)

        derived_params_collection = self.likelihood.get_derived_parameters()
        assert derived_params_collection is not None
        sample.put_double(section_names.likelihoods, "firecrown_like", loglike)
        for section, name, val in derived_params_collection:
            sample.put(section, name, val)

        if not isinstance(self.likelihood, GaussFamily):
            self.likelihood.reset()
            self.tools.reset()
            return 0

        # If we get here, we have a GaussFamily likelihood, and we need to
        # save concatenated data vector and inverse covariance to enable support
        # for the CosmoSIS Fisher sampler. This can only work for likelihoods
        # that have these quantities. Currently, this is only GaussFamily.

        sample.put(
            "data_vector",
            "firecrown_theory",
            self.likelihood.get_theory_vector(),
        )
        sample.put(
            "data_vector",
            "firecrown_data",
            self.likelihood.get_data_vector(),
        )
        sample.put(
            "data_vector",
            "firecrown_inverse_covariance",
            self.likelihood.inv_cov,
        )

        # Write out theory and data vectors to the data block the ease
        # debugging.
        # TODO: This logic should be moved into the TwoPoint statistic, and
        # some method in the Statistic base class should be called here. For
        # statistics other than TwoPoint, the base class implementation should
        # do nothing.
        for stat in self.likelihood.statistics:
            if isinstance(stat, TwoPoint):
                assert stat.sacc_tracers is not None
                tracer = f"{stat.sacc_tracers[0]}_{stat.sacc_tracers[1]}"

                sample.put(
                    "data_vector",
                    f"ell_or_theta_{stat.sacc_data_type}_{tracer}",
                    stat.ell_or_theta_,
                )
                sample.put(
                    "data_vector",
                    f"theory_{stat.sacc_data_type}_{tracer}",
                    stat.get_theory_vector(),
                )
                sample.put(
                    "data_vector",
                    f"data_{stat.sacc_data_type}_{tracer}",
                    stat.get_data_vector(),
                )

        self.likelihood.reset()
        self.tools.reset()
        return 0

    def form_error_message(self, exc: MissingSamplerParameterError) -> str:
        """Form the error message that will be used to report a missing parameter.

        This error message will also include when that parameter should have been
        supplied by the sampler.
        """
        msg = (
            "A required parameter was not found in any of the "
            "sections searched on DataBlock.\n"
            "These are specified by the space-separated string "
            "`sampling_parameter_sections`.\n"
            "The supplied value was"
        )
        sampling_parameters_sections = " ".join(self.sampling_sections)
        if sampling_parameters_sections:
            msg += f": `{sampling_parameters_sections}`"
        else:
            msg += " an empty string."
        msg += f"\nThe missing parameter is named: `{exc.parameter}`\n"
        return msg

    def calculate_firecrown_params(self, sample: cosmosis.datablock) -> ParamsMap:
        """Calculate the ParamsMap for this sample."""
        firecrown_params = ParamsMap()
        for section in self.sampling_sections:
            section_params = extract_section(sample, section)
            shared_keys = section_params.to_set().intersection(firecrown_params)
            if len(shared_keys) > 0:
                raise RuntimeError(
                    f"The following keys `{shared_keys}' appear "
                    f"in more than one section used by the "
                    f"module {self.firecrown_module_name}."
                )

            firecrown_params = ParamsMap({**firecrown_params, **section_params.data})

        firecrown_params.use_lower_case_keys(True)
        return firecrown_params


def setup(config: cosmosis.datablock) -> FirecrownLikelihood:
    """Setup hook for a CosmoSIS module.

    Returns an instance of
    class FirecrownLikelihood. The same object will be passed to the CosmoSIS
    execute hook.
    """
    return FirecrownLikelihood(config)


def execute(sample: cosmosis.datablock, instance: FirecrownLikelihood) -> int:
    """Execute hook for a CosmoSIS module.

    Return 0 on success. The parameter `sample` represents the current MCMC sample;
    `instance` is the FirecrownLikelihood object created by `setup`.
    """
    return instance.execute(sample)


def cleanup(_) -> int:
    """Cleanup hook for a CosmoSIS module. This one has nothing to do."""
    return 0
