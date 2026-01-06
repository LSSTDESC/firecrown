import pytest

from cosmosis.datablock import DataBlock, option_section

from firecrown.connector.cosmosis.likelihood import FirecrownLikelihood


def make_config_for_pure_ccl() -> DataBlock:
    cfg = DataBlock()
    cfg.put_string(
        option_section,
        "likelihood_source",
        "tests/likelihood/lkdir/lkscript_pure_ccl.py",
    )
    # sampling parameters will be read from this section by calculate_firecrown_params
    cfg.put_string(
        option_section, "sampling_parameters_sections", "firecrown_two_point"
    )
    return cfg


def test_execute_conflict_raises_runtime_error():
    cfg = make_config_for_pure_ccl()
    module = FirecrownLikelihood(cfg)

    # The EmptyLikelihood used in this test does not consume sampling-specific
    # parameters; disable raising on unused keys so that extra sampling keys
    # do not cause the test to fail.
    module.likelihood.raise_on_unused_parameter = False

    sample = DataBlock()
    # cosmological_parameters present (cosmo keys in CosmoSIS style)
    sample.put_double("cosmological_parameters", "omega_c", 0.25)
    sample.put_double("cosmological_parameters", "omega_b", 0.05)
    sample.put_double("cosmological_parameters", "h0", 0.67)
    sample.put_double("cosmological_parameters", "n_s", 0.96)
    sample.put_double("cosmological_parameters", "omega_k", 0.0)
    sample.put_double("cosmological_parameters", "nnu", 3.044)
    sample.put_double("cosmological_parameters", "mnu", 0.0)
    sample.put_double("cosmological_parameters", "w0", -1.0)
    sample.put_double("cosmological_parameters", "wa", 0.0)
    sample.put_double("cosmological_parameters", "tcmb", 2.7255)
    sample.put_double("cosmological_parameters", "sigma_8", 0.81)

    # sampling section contains same canonical key but different value => conflict
    sample.put_double("firecrown_two_point", "Omega_c", 0.26)

    with pytest.raises(RuntimeError, match="Conflicting cosmological parameter"):
        module.execute(sample)


def test_execute_successful_merge_and_run():
    cfg = make_config_for_pure_ccl()
    module = FirecrownLikelihood(cfg)

    # The EmptyLikelihood used in this test does not consume sampling-specific
    # parameters; disable raising on unused keys so that extra sampling keys
    # do not cause the test to fail.
    module.likelihood.raise_on_unused_parameter = False

    sample = DataBlock()
    # supply cosmology via cosmological_parameters using CosmoSIS names
    sample.put_double("cosmological_parameters", "omega_c", 0.25)
    sample.put_double("cosmological_parameters", "omega_b", 0.05)
    sample.put_double("cosmological_parameters", "h0", 0.67)
    sample.put_double("cosmological_parameters", "n_s", 0.96)
    sample.put_double("cosmological_parameters", "omega_k", 0.0)
    sample.put_double("cosmological_parameters", "nnu", 3.044)
    sample.put_double("cosmological_parameters", "mnu", 0.0)
    sample.put_double("cosmological_parameters", "w0", -1.0)
    sample.put_double("cosmological_parameters", "wa", 0.0)
    sample.put_double("cosmological_parameters", "tcmb", 2.7255)
    sample.put_double("cosmological_parameters", "sigma_8", 0.81)
    # ensure sampling section does not conflict (no Omega_c present)
    # include a sampling key that will be consumed by CCLFactory so the section
    # exists but does not introduce unused-parameter errors
    sample.put_double("firecrown_two_point", "Omega_b", 0.05)

    status = module.execute(sample)
    assert status == 0
    # verify the likelihood value was written
    assert sample.has_section("likelihoods")
    # the exact key written is 'firecrown_like'
    assert sample.has_value("likelihoods", "firecrown_like")
