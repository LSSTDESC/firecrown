"""Unit testing for firecrown's CosmoSIS module.

As a unit test, what this can test is very limited.
This test do not invoke the `cosmosis` executable.
"""

from os.path import expandvars
import yaml
import pytest
import numpy as np

from cosmosis.datablock import DataBlock, option_section, names as section_names

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.connector.cosmosis.likelihood import (
    FirecrownLikelihood,
    extract_section,
    MissingSamplerParameterError,
    execute,
    form_error_message,
)


@pytest.fixture(name="minimal_module_config")
def fixture_minimal_module_config() -> DataBlock:
    """Return a minimal CosmoSIS datablock.
    It contains only the module's filename.
    This is the minimal possible configuration."""
    block = DataBlock()
    block.put_string(
        option_section, "likelihood_source", "tests/likelihood/lkdir/lkscript.py"
    )
    return block


@pytest.fixture(name="defective_module_config")
def fixture_defective_module_config() -> DataBlock:
    """Return a CosmoSIS datablock that lacks the required
    parameter to configure a ParameterizedLikelihood."""
    block = DataBlock()
    block.put_string(
        option_section,
        "likelihood_source",
        expandvars("${FIRECROWN_DIR}/tests/likelihood/lkdir/lk_needing_param.py"),
    )
    return block


@pytest.fixture(name="minimal_config")
def fixture_minimal_config() -> DataBlock:
    result = DataBlock()
    result.put_string(
        option_section,
        "likelihood_source",
        expandvars("${FIRECROWN_DIR}/tests/likelihood/lkdir/lkscript.py"),
    )
    return result


@pytest.fixture(name="config_with_derived_parameters")
def fixture_config_with_derived_parameters() -> DataBlock:
    result = DataBlock()
    result.put_string(
        option_section,
        "Likelihood_source",
        expandvars("${FIRECROWN_DIR}/tests/likelihood/lkdir/lk_derived_parameter.py"),
    )
    return result


@pytest.fixture(name="config_with_const_gaussian_missing_sampling_parameters_sections")
def fixture_config_with_const_gaussian_missing_sampling_parameters_sections() -> (
    DataBlock
):
    result = DataBlock()
    result.put_string(
        option_section,
        "Likelihood_source",
        expandvars(
            "${FIRECROWN_DIR}/tests/likelihood/gauss_family/lkscript_const_gaussian.py"
        ),
    )
    # result.put_string(option_section, )
    return result


@pytest.fixture(name="config_with_two_point_real")
def fixture_config_with_two_point_real() -> DataBlock:
    result = DataBlock()
    result.put_string(
        option_section,
        "Likelihood_source",
        expandvars(
            "${FIRECROWN_DIR}/tests/likelihood/gauss_family/lkscript_two_point.py"
        ),
    )
    result.put_string(
        option_section,
        "sampling_parameters_sections",
        "firecrown_two_point_parameters",
    )
    result.put_string(
        option_section,
        "projection",
        "real",
    )
    return result


@pytest.fixture(name="config_with_two_point_harmonic", params=["default", "pure_ccl"])
def fixture_config_with_two_point_harmonic(request) -> DataBlock:
    result = DataBlock()
    likelihood_file = ""
    if request.param == "default":
        likelihood_file = (
            "${FIRECROWN_DIR}/tests/likelihood/gauss_family/lkscript_two_point.py"
        )
    elif request.param == "pure_ccl":
        likelihood_file = (
            "${FIRECROWN_DIR}/tests/likelihood/"
            "gauss_family/lkscript_two_point_pure_ccl.py"
        )

    result.put_string(
        option_section,
        "Likelihood_source",
        expandvars(likelihood_file),
    )
    result.put_string(
        option_section,
        "sampling_parameters_sections",
        "firecrown_two_point_parameters",
    )
    result.put_string(
        option_section,
        "projection",
        "harmonic",
    )
    return result


@pytest.fixture(name="minimal_firecrown_mod")
def fixture_minimal_firecrown_mod(minimal_config: DataBlock) -> FirecrownLikelihood:
    return FirecrownLikelihood(minimal_config)


@pytest.fixture(name="firecrown_mod_with_derived_parameters")
def fixture_firecrown_mod_with_derived_parameters(
    config_with_derived_parameters: DataBlock,
) -> FirecrownLikelihood:
    return FirecrownLikelihood(config_with_derived_parameters)


@pytest.fixture(name="firecrown_mod_with_const_gaussian")
def fixture_firecrown_mod_with_const_gaussian(
    working_config_for_const_gaussian: DataBlock,
) -> FirecrownLikelihood:
    result = FirecrownLikelihood(working_config_for_const_gaussian)
    return result


@pytest.fixture(name="firecrown_mod_with_two_point_real")
def fixture_firecrown_mod_with_two_point_real(
    config_with_two_point_real: DataBlock,
) -> FirecrownLikelihood:
    result = FirecrownLikelihood(config_with_two_point_real)
    return result


@pytest.fixture(name="firecrown_mod_with_two_point_harmonic")
def fixture_firecrown_mod_with_two_point_harmonic(
    config_with_two_point_harmonic: DataBlock,
) -> FirecrownLikelihood:
    result = FirecrownLikelihood(config_with_two_point_harmonic)
    return result


@pytest.fixture(name="sample_with_cosmo")
def fixture_sample_with_cosmo() -> DataBlock:
    """Return a DataBlock that contains some cosmological parameters."""
    result = DataBlock()
    params = {
        "h0": 0.83,
        "omega_b": 0.22,
        "omega_c": 0.120,
        "omega_k": 0.0,
        "omega_nu": 0.0,
        "w": -1.0,
        "wa": 0.0,
    }
    for name, val in params.items():
        result.put_double("cosmological_parameters", name, val)
    return result


@pytest.fixture(name="minimal_sample")
def fixture_minimal_sample(sample_with_cosmo: DataBlock) -> DataBlock:
    with open("tests/distances.yml", encoding="utf-8") as stream:
        rawdata = yaml.load(stream, yaml.CLoader)
    sample = sample_with_cosmo
    for section_name, section_data in rawdata.items():
        for parameter_name, value in section_data.items():
            sample.put(section_name, parameter_name, np.array(value))
    return sample


@pytest.fixture(name="minimal_sample_with_pk")
def fixture_minimal_sample_with_pk(sample_with_cosmo: DataBlock) -> DataBlock:
    with open("tests/distances_and_pk.yml", encoding="utf-8") as stream:
        rawdata = yaml.load(stream, yaml.CLoader)
    sample = sample_with_cosmo
    for section_name, section_data in rawdata.items():
        for parameter_name, value in section_data.items():
            sample.put(section_name, parameter_name, np.array(value))
    return sample


@pytest.fixture(name="sample_with_M")
def fixture_sample_with_M(minimal_sample: DataBlock) -> DataBlock:
    minimal_sample.put("supernova_parameters", "pantheon_M", 4.5)
    return minimal_sample


@pytest.fixture(name="sample_with_lens0_bias")
def fixture_sample_with_lens0_bias(minimal_sample_with_pk: DataBlock) -> DataBlock:
    minimal_sample_with_pk.put("firecrown_two_point_parameters", "lens0_bias", 1.0)
    return minimal_sample_with_pk


@pytest.fixture(name="sample_without_M")
def fixture_sample_without_M(minimal_sample: DataBlock) -> DataBlock:
    minimal_sample.put("supernova_parameters", "nobody_loves_me", 4.5)
    return minimal_sample


def test_extract_section_gets_named_parameters(defective_module_config: DataBlock):
    params = extract_section(defective_module_config, "module_options")
    assert isinstance(params, NamedParameters)
    assert params.get_string("likelihood_source") == expandvars(
        "${FIRECROWN_DIR}/tests/likelihood/lkdir/lk_needing_param.py"
    )


def test_extract_section_raises_on_missing_section(defective_module_config: DataBlock):
    with pytest.raises(RuntimeError, match="Datablock section `xxx' does not exist"):
        _ = extract_section(defective_module_config, "xxx")


def test_parameterless_module_construction(minimal_module_config):
    """Make sure we can create a CosmoSIS likelihood modules that does not
    introduce any new parameters."""
    module = FirecrownLikelihood(minimal_module_config)
    assert module.sampling_sections == []


def test_missing_required_parameter(defective_module_config):
    """Make sure that a missing required parameter entails the expected
    failure."""
    with pytest.raises(KeyError):
        _ = FirecrownLikelihood(defective_module_config)


def test_initialize_minimal_module(minimal_firecrown_mod: FirecrownLikelihood):
    assert isinstance(minimal_firecrown_mod, FirecrownLikelihood)


def test_execute_missing_cosmological_parameters(
    minimal_firecrown_mod: FirecrownLikelihood,
):
    no_cosmo_params = DataBlock()
    with pytest.raises(
        RuntimeError,
        match="Datablock section `cosmological_parameters' does not exist.",
    ):
        _ = minimal_firecrown_mod.execute(no_cosmo_params)


def test_execute_with_cosmo(
    minimal_firecrown_mod: FirecrownLikelihood, minimal_sample: DataBlock
):
    assert minimal_firecrown_mod.execute(minimal_sample) == 0
    assert minimal_sample[section_names.likelihoods, "firecrown_like"] == -3.0


def test_execute_with_derived_parameters(
    firecrown_mod_with_derived_parameters: FirecrownLikelihood,
    minimal_sample: DataBlock,
):
    assert firecrown_mod_with_derived_parameters.execute(minimal_sample) == 0
    assert minimal_sample.get_double("derived_section", "derived_param0") == 1.0


def test_module_init_with_missing_sampling_sections(
    config_with_const_gaussian_missing_sampling_parameters_sections: DataBlock,
):
    with pytest.raises(RuntimeError, match=r"\['pantheon_M'\]"):
        s = config_with_const_gaussian_missing_sampling_parameters_sections.to_string()
        assert s is not None
        _ = FirecrownLikelihood(
            config_with_const_gaussian_missing_sampling_parameters_sections
        )


@pytest.fixture(name="config_with_wrong_sampling_sections")
def fixture_config_with_wrong_sampling_sections(
    config_with_const_gaussian_missing_sampling_parameters_sections: DataBlock,
) -> DataBlock:
    # The giant name is good documentation, but I can't type that correctly
    # twice.
    config = config_with_const_gaussian_missing_sampling_parameters_sections
    config[option_section, "sampling_parameters_sections"] = "this_is_wrong"
    return config


@pytest.fixture(name="working_config_for_const_gaussian")
def fixture_working_config_for_const_gaussian(
    config_with_const_gaussian_missing_sampling_parameters_sections: DataBlock,
) -> DataBlock:
    config = config_with_const_gaussian_missing_sampling_parameters_sections
    config[option_section, "sampling_parameters_sections"] = "supernova_parameters"
    return config


def test_module_init_with_wrong_sampling_sections(
    config_with_wrong_sampling_sections: DataBlock,
):
    mod = FirecrownLikelihood(config_with_wrong_sampling_sections)
    assert isinstance(mod, FirecrownLikelihood)


def test_module_exec_with_wrong_sampling_sections(
    config_with_wrong_sampling_sections: DataBlock, sample_with_M: DataBlock
):
    mod = FirecrownLikelihood(config_with_wrong_sampling_sections)
    with pytest.raises(
        RuntimeError, match="Datablock section `this_is_wrong' does not exist"
    ):
        _ = mod.execute(sample_with_M)


def test_module_exec_missing_parameter_in_sampling_sections(
    firecrown_mod_with_const_gaussian: FirecrownLikelihood, sample_without_M: DataBlock
):
    with pytest.raises(RuntimeError, match="`supernova_parameters`") as exc:
        _ = firecrown_mod_with_const_gaussian.execute(sample_without_M)
    outer_execption = exc.value
    inner_exception = outer_execption.__cause__
    assert isinstance(inner_exception, MissingSamplerParameterError)
    assert inner_exception.parameter == "pantheon_M"


def test_module_exec_working(
    firecrown_mod_with_const_gaussian: FirecrownLikelihood, sample_with_M: DataBlock
):
    assert firecrown_mod_with_const_gaussian.execute(sample_with_M) == 0
    assert sample_with_M.get_double("likelihoods", "firecrown_like") < 0.0


def test_execute_function(
    firecrown_mod_with_const_gaussian: FirecrownLikelihood, sample_with_M: DataBlock
):
    assert execute(sample_with_M, firecrown_mod_with_const_gaussian) == 0
    assert sample_with_M.get_double("likelihoods", "firecrown_like") < 0.0


def test_module_exec_with_two_point_real(
    firecrown_mod_with_two_point_real: FirecrownLikelihood,
    sample_with_lens0_bias: DataBlock,
):
    assert firecrown_mod_with_two_point_real.execute(sample_with_lens0_bias) == 0

    # CosmoSIS always writes the output to the same section, so we can
    # check if the connector is writing the expected values.
    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d(
                "data_vector", "firecrown_theory"
            )
        )
    )
    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d("data_vector", "firecrown_data")
        )
    )
    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_nd(
                "data_vector", "firecrown_inverse_covariance"
            )
        )
    )

    # When dealing with a two-point statistic, the connector should write
    # the related quantities to the datablock.
    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d(
                "data_vector", "theta_galaxy_density_xi_lens0_lens0"
            )
        )
    )

    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d(
                "data_vector", "theory_galaxy_density_xi_lens0_lens0"
            )
        )
    )

    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d(
                "data_vector", "data_galaxy_density_xi_lens0_lens0"
            )
        )
    )


def test_module_exec_with_two_point_harmonic(
    firecrown_mod_with_two_point_harmonic: FirecrownLikelihood,
    sample_with_lens0_bias: DataBlock,
):
    assert firecrown_mod_with_two_point_harmonic.execute(sample_with_lens0_bias) == 0

    # CosmoSIS always writes the output to the same section, so we can
    # check if the connector is writing the expected values.
    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d(
                "data_vector", "firecrown_theory"
            )
        )
    )
    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d("data_vector", "firecrown_data")
        )
    )
    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_nd(
                "data_vector", "firecrown_inverse_covariance"
            )
        )
    )

    # When dealing with a two-point statistic, the connector should write
    # the related quantities to the datablock.
    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_int_array_1d(
                "data_vector", "ell_galaxy_density_cl_lens0_lens0"
            )
        )
    )

    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d(
                "data_vector", "theory_galaxy_density_cl_lens0_lens0"
            )
        )
    )

    assert np.all(
        np.isfinite(
            sample_with_lens0_bias.get_double_array_1d(
                "data_vector", "data_galaxy_density_cl_lens0_lens0"
            )
        )
    )


def test_mapping_cosmosis_background(mapping_cosmosis):
    block = DataBlock()

    background_d_m = np.geomspace(0.1, 10.0, 100)
    background_z = np.linspace(0.0, 2.0, 10)
    background_h = np.geomspace(0.1, 10.0, 100)

    block.put_double_array_1d("distances", "d_m", background_d_m)
    block.put_double_array_1d("distances", "z", background_z)
    block.put_double_array_1d("distances", "h", background_h)

    ccl_args = mapping_cosmosis.calculate_ccl_args(block)

    assert "background" in ccl_args
    assert np.allclose(
        ccl_args["background"]["a"],
        mapping_cosmosis.redshift_to_scale_factor(background_z),
    )
    assert np.allclose(ccl_args["background"]["chi"], np.flip(background_d_m))
    assert np.allclose(
        ccl_args["background"]["h_over_h0"],
        mapping_cosmosis.transform_h_to_h_over_h0(background_h),
    )

    assert "pk_linear" not in ccl_args
    assert "pk_nonlin" not in ccl_args


def test_mapping_cosmosis_pk_linear(mapping_cosmosis):
    block = DataBlock()

    block.put_double_array_1d("distances", "d_m", np.geomspace(0.1, 10.0, 100))
    block.put_double_array_1d("distances", "z", np.linspace(0.0, 2.0, 10))
    block.put_double_array_1d("distances", "h", np.geomspace(0.1, 10.0, 100))

    matter_power_lin_k_h = np.geomspace(0.1, 10.0, 100)
    matter_power_lin_z = np.linspace(0.0, 2.0, 10)
    matter_power_lin_p_k = np.geomspace(1.0e-3, 1.0e3, 100)

    block.put_double_array_1d("matter_power_lin", "k_h", matter_power_lin_k_h)
    block.put_double_array_1d("matter_power_lin", "z", matter_power_lin_z)
    block.put_double_array_1d("matter_power_lin", "p_k", matter_power_lin_p_k)

    ccl_args = mapping_cosmosis.calculate_ccl_args(block)

    assert "background" in ccl_args
    assert "pk_linear" in ccl_args
    assert "pk_nonlin" not in ccl_args

    assert np.allclose(
        ccl_args["pk_linear"]["a"],
        mapping_cosmosis.redshift_to_scale_factor(matter_power_lin_z),
    )
    assert np.allclose(
        ccl_args["pk_linear"]["k"],
        mapping_cosmosis.transform_k_h_to_k(matter_power_lin_k_h),
    )
    assert np.allclose(
        ccl_args["pk_linear"]["delta_matter:delta_matter"],
        mapping_cosmosis.redshift_to_scale_factor_p_k(
            mapping_cosmosis.transform_p_k_h3_to_p_k(matter_power_lin_p_k)
        ),
    )


def test_mapping_cosmosis_pk_nonlin(mapping_cosmosis):
    block = DataBlock()

    block.put_double_array_1d("distances", "d_m", np.geomspace(0.1, 10.0, 100))
    block.put_double_array_1d("distances", "z", np.linspace(0.0, 2.0, 10))
    block.put_double_array_1d("distances", "h", np.geomspace(0.1, 10.0, 100))

    matter_power_nl_k_h = np.geomspace(0.1, 10.0, 100)
    matter_power_nl_z = np.linspace(0.0, 2.0, 10)
    matter_power_nl_p_k = np.geomspace(1.0e-3, 1.0e3, 100)

    block.put_double_array_1d("matter_power_nl", "k_h", matter_power_nl_k_h)
    block.put_double_array_1d("matter_power_nl", "z", matter_power_nl_z)
    block.put_double_array_1d("matter_power_nl", "p_k", matter_power_nl_p_k)

    ccl_args = mapping_cosmosis.calculate_ccl_args(block)

    assert "background" in ccl_args
    assert "pk_linear" not in ccl_args
    assert "pk_nonlin" in ccl_args
    assert "nonlinear_model" not in ccl_args

    assert np.allclose(
        ccl_args["pk_nonlin"]["a"],
        mapping_cosmosis.redshift_to_scale_factor(matter_power_nl_z),
    )
    assert np.allclose(
        ccl_args["pk_nonlin"]["k"],
        mapping_cosmosis.transform_k_h_to_k(matter_power_nl_k_h),
    )
    assert np.allclose(
        ccl_args["pk_nonlin"]["delta_matter:delta_matter"],
        mapping_cosmosis.redshift_to_scale_factor_p_k(
            mapping_cosmosis.transform_p_k_h3_to_p_k(matter_power_nl_p_k)
        ),
    )


def test_form_error_message_with_sampling_sections():
    sampling_sections = ["section1", "section2"]
    exc = MissingSamplerParameterError("missing_param")
    expected_msg = (
        "A required parameter was not found in any of the sections searched on DataBlock.\n"  # noqa
        "These are specified by the space-separated string `sampling_parameter_sections`.\n"  # noqa
        "The supplied value was: `section1 section2`\n"
        "The missing parameter is named: `missing_param`\n"
    )
    assert form_error_message(sampling_sections, exc) == expected_msg


def test_form_error_message_with_empty_sampling_sections():
    sampling_sections: list[str] = []
    exc = MissingSamplerParameterError("missing_param")
    #  flake8: noqa
    expected_msg = (
        "A required parameter was not found in any of the sections searched on DataBlock.\n"  # noqa
        "These are specified by the space-separated string `sampling_parameter_sections`.\n"  # noqa
        "The supplied value was an empty string.\n"
        "The missing parameter is named: `missing_param`\n"
    )
    assert form_error_message(sampling_sections, exc) == expected_msg


def test_form_error_message_with_single_sampling_section():
    sampling_sections = ["section1"]
    exc = MissingSamplerParameterError("missing_param")
    #  flake8: noqa
    expected_msg = (
        "A required parameter was not found in any of the sections searched on DataBlock.\n"  # noqa
        "These are specified by the space-separated string `sampling_parameter_sections`.\n"  # noqa
        "The supplied value was: `section1`\n"
        "The missing parameter is named: `missing_param`\n"
    )
    assert form_error_message(sampling_sections, exc) == expected_msg


def test_form_error_message_with_missing_sampler_parameter_error():
    sampling_sections = ["section1", "section2"]
    exc = MissingSamplerParameterError("missing_param")
    assert "missing_param" in form_error_message(sampling_sections, exc)
