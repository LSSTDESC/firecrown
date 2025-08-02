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

from firecrown.connector.mapping import mapping_builder, MappingCosmoSIS
from firecrown.likelihood.gaussfamily import GaussFamily
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.likelihood import load_likelihood, Likelihood, NamedParameters
from firecrown.ccl_factory import CCLCreationMode
from firecrown.parameters import ParamsMap, handle_unused_params
from firecrown.updatable import MissingSamplerParameterError


def extract_section(sample: cosmosis.datablock, section: str) -> NamedParameters:
    """Extract all the parameters from the name datablock section into a dictionary.

    :param sample: the CosmoSiS datablock to query
    :param section: the name of the section desired
    :return: a dictionary of the parameters in the section
    """
    if not sample.has_section(section):
        raise RuntimeError(f"Datablock section `{section}' does not exist.")
    sec_dict = {name: sample[section, name] for _, name in sample.keys(section=section)}
    return NamedParameters(sec_dict)


class FirecrownLikelihood:
    """CosmoSIS likelihood module for calculating a Firecrown likelihood.

    In this simplest implementation, we have only a single module. This module
    is responsible for calling CCL to perform theory calculations, based on the
    output of CAMB, and also for calculating the data likelihood based on this
    theory.
    """

    def __init__(self, config: cosmosis.datablock) -> None:
        """Create the FirecrownLikelihood object from the given configuration.

        :param config: the datablock the configuration
        """
        likelihood_source = config.get_string(option_section, "likelihood_source", "")
        if likelihood_source == "":
            likelihood_source = config[option_section, "firecrown_config"]

        build_parameters = extract_section(config, option_section)

        sections_str: str = config.get_string(
            option_section, "sampling_parameters_sections", ""
        )
        assert isinstance(sections_str, str)
        sections = sections_str.split()

        self.firecrown_module_name = option_section
        self.sampling_sections: list[str] = sections
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
        # We have to do some extra type-fiddling here because mapping_builder
        # has a declared return type of the base class.
        new_mapping = mapping_builder(input_style="CosmoSIS")
        assert isinstance(new_mapping, MappingCosmoSIS)
        self.map = new_mapping

        # If sampling_sections  is empty, but we have required parameters, then
        # we have a configuration problem, and ParamsMap can never be built
        # correctly.
        if len(self.sampling_sections) == 0:
            required_parameters = (
                self.likelihood.required_parameters() + self.tools.required_parameters()
            )
            required_parameters -= self.tools.ccl_factory.required_parameters()

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
        # If we are using CCL to calculate our cosmology, then CosmoSIS should
        # not also be configured to have its CAMB module calculate the cosmology.
        # The DEFAULT mode is the only one that uses the CAMB module to calculate
        # the cosmology.
        if self.tools.ccl_factory.creation_mode != CCLCreationMode.DEFAULT:
            if config.has_section("camb") and (
                not self.tools.ccl_factory.allow_multiple_camb_instances
            ):
                raise RuntimeError(
                    "If Firecrown is using CCL to calculate the cosmology, then "
                    "CosmoSIS should not be configured to use CAMB to "
                    "calculate the cosmology."
                )

    def execute(self, sample: cosmosis.datablock) -> int:
        """This is the method called for each sample generated by the sampler.

        :param sample: the sample generated by the sampler
        :return: 0
        """
        cosmological_params: NamedParameters = extract_section(
            sample, "cosmological_parameters"
        )
        self.map.set_params_from_cosmosis(cosmological_params)

        # TODO: Future development will need to capture elements that get put into the
        # datablock. This probably will be in a different "physics module" and not in
        # the likelihood module. And it requires updates to Firecrown to split the
        # calculations. e.g., data_vector/firecrown_theory  data_vector/firecrown_data

        firecrown_params = calculate_firecrown_params(
            self.sampling_sections, self.firecrown_module_name, sample
        )
        firecrown_params = ParamsMap(firecrown_params | self.map.asdict())
        firecrown_params.use_lower_case_keys(True)
        self.update_likelihood_and_tools(firecrown_params)

        if self.tools.ccl_factory.creation_mode == CCLCreationMode.DEFAULT:
            # Cosmology will be read from datablock
            self.tools.prepare(calculator_args=self.map.calculate_ccl_args(sample))
        else:
            # Cosmology will be generated by CCL. We want to be sure that CosmoSIS
            # has not also been configured to use CAMB.
            self.tools.prepare()

        handle_unused_params(params=firecrown_params, raise_on_unused=False)
        loglike = self.likelihood.compute_loglike_for_sampling(self.tools)

        derived_params_collection = self.likelihood.get_derived_parameters()
        assert derived_params_collection is not None
        sample.put_double(section_names.likelihoods, "firecrown_like", loglike)
        for section, name, val in derived_params_collection:
            sample.put(section, name, val)

        if not isinstance(self.likelihood, GaussFamily):
            self.likelihood.reset()
            self.tools.reset()
            return 0

        self.special_gauss_family_handling(sample)

        self.likelihood.reset()
        self.tools.reset()
        return 0

    def special_gauss_family_handling(self, sample: cosmosis.datablock) -> None:
        """Special handling for the GaussFamily likelihood.

        We need to save concatenated data vector and inverse covariance to
        enable support for the CosmoSIS Fisher sampler. This can only work
        for likelihoods that have these quantities. Currently, this is only
        GaussFamily.

        :param sample: the sample generated by the sampler
        :return: None
        """
        assert isinstance(self.likelihood, GaussFamily)
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
        for gstat in self.likelihood.statistics:
            stat = gstat.statistic

            if isinstance(stat, TwoPoint):
                self.handle_twopoint_statistic(sample, stat)

    def handle_twopoint_statistic(
        self, sample: cosmosis.datablock, stat: TwoPoint
    ) -> None:
        """Handle the TwoPoint statistic for the GaussFamily likelihood.

        This puts the theory and data vectors in the data block.

        :param sample: the sample generated by the sampler
        :param stat: a TwoPoint statistic
        """
        assert stat.sacc_tracers is not None
        tracer = f"{stat.sacc_tracers[0]}_{stat.sacc_tracers[1]}"

        # Determine whether we have ells or thetas -- it must not be both
        have_ells = stat.ells is not None
        have_thetas = stat.thetas is not None
        assert have_ells != have_thetas

        if have_ells:
            sample.put_int_array_1d(
                "data_vector",
                f"ell_{stat.sacc_data_type}_{tracer}",
                stat.ells,
            )

        if have_thetas:
            sample.put_double_array_1d(
                "data_vector",
                f"theta_{stat.sacc_data_type}_{tracer}",
                stat.thetas,
            )

        sample.put_double_array_1d(
            "data_vector",
            f"theory_{stat.sacc_data_type}_{tracer}",
            stat.get_theory_vector(),
        )
        sample.put_double_array_1d(
            "data_vector",
            f"data_{stat.sacc_data_type}_{tracer}",
            stat.get_data_vector(),
        )

    def update_likelihood_and_tools(self, firecrown_params: ParamsMap) -> None:
        """Update the likelihood and tools with the new parameters.

        :param firecrown_params: the new parameters
        :return: None
        """
        try:
            self.likelihood.update(firecrown_params)
            self.tools.update(firecrown_params)
        except MissingSamplerParameterError as exc:
            msg = form_error_message(self.sampling_sections, exc)
            raise RuntimeError(msg) from exc


def calculate_firecrown_params(
    sampling_sections: list[str], firecrown_module_name: str, sample: cosmosis.datablock
) -> ParamsMap:
    """Calculate the ParamsMap for this sample.

    :param sample: the sample generated by the sampler
    :return: a ParamsMap with the firecrown parameters
    """
    firecrown_params = ParamsMap()
    for section in sampling_sections:
        section_params = extract_section(sample, section)
        shared_keys = section_params.to_set().intersection(firecrown_params)
        if len(shared_keys) > 0:
            raise RuntimeError(
                f"The following keys `{shared_keys}` appear "
                f"in more than one section used by the "
                f"module {firecrown_module_name}."
            )

        firecrown_params = ParamsMap({**firecrown_params, **section_params.data})

    firecrown_params.use_lower_case_keys(True)
    return firecrown_params


def form_error_message(
    sampling_sections: list[str], exc: MissingSamplerParameterError
) -> str:
    """Form the error message that will be used to report a missing parameter.

    This error message will also include when that parameter should have been
    supplied by the sampler.

    :param exc: the missing parameter error
    :return: the error message
    """
    msg = (
        "A required parameter was not found in any of the "
        "sections searched on DataBlock.\n"
        "These are specified by the space-separated string "
        "`sampling_parameter_sections`.\n"
        "The supplied value was"
    )
    sampling_parameters_sections = " ".join(sampling_sections)
    if sampling_parameters_sections:
        msg += f": `{sampling_parameters_sections}`"
    else:
        msg += " an empty string."
    msg += f"\nThe missing parameter is named: `{exc.parameter}`\n"
    return msg


def setup(config: cosmosis.datablock) -> FirecrownLikelihood:
    """Setup hook for a CosmoSIS module.

    The returned object will be passed to the CosmoSIS execute hook.

    :param config: the datablock the configuration
    :return: an instance of class FirecrownLikelihood
    """
    return FirecrownLikelihood(config)


def execute(sample: cosmosis.datablock, instance: FirecrownLikelihood) -> int:
    """Execute hook for a CosmoSIS module.

    Return 0 on success. The parameter `sample` represents the current MCMC sample;
    `instance` is the FirecrownLikelihood object created by `setup`.

    :param sample: the sample generated by the sampler
    :param instance: the FirecrownLikelihood object
    :return: the status of the call to the module's execute function
    """
    return instance.execute(sample)


def cleanup(_) -> int:
    """Cleanup hook for a CosmoSIS module. This one has nothing to do.

    :return: 0
    """
    return 0
