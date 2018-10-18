import pytest
from ..cosmosis.run import _make_parallel_pool, _make_cosmosis_config
from ..cosmosis.run import _make_cosmosis_values, _make_cosmosis_pipeline
import yaml
import numpy as np

try:
    import cosmosis
except ImportError:
    cosmosis = None

requires_cosmosis = pytest.mark.skipif(cosmosis is None,
                                       reason="cosmosis not installed")


@pytest.fixture(scope="session")
def tx_config(tmpdir_factory):
    config_text = """
parameters:
  Omega_k: 0.0
  Omega_c: [0.25, 0.27, 0.32]  # [min, start, max]
  Omega_b: 0.045
  h: 0.67
  n_s: 0.96
  A_s: [2.0e-9, 2.1e-9, 2.2e-9]
  w0: -1.0
  wa: 0.0

  # lens bin zero
  src0_delta_z: 0.0
  src1_delta_z: 0.0


sampler:
  sampler: test
  output: chain.txt
  debug: True
  quiet: False
  mpi: False
  # parameters for individual samplers:
  test:
    # pretending this sampler can take all these options
    fatal_errors: True
    walkers: 10
    nsample: 20
    nsample_dimension: 5
    step_size: 0.02
"""
    config = yaml.load(config_text)
    return config


@requires_cosmosis
def test_pool(tx_config):
    pool = _make_parallel_pool(tx_config['sampler'])
    assert pool is None or pool.size > 0
    assert pool is None or isinstance(pool, cosmosis.runtime.mpi_pool.MPIPool)

    # In general I'm wary of attempting tests that
    # need MPI - bit of a minefield.
    # Advice welcome.


@requires_cosmosis
def test_config(tx_config):
    ini = _make_cosmosis_config(tx_config['sampler'])
    assert isinstance(ini, cosmosis.runtime.config.Inifile)
    assert ini.getint('test', 'walkers') == 10
    assert np.isclose(ini.getfloat('test', 'step_size'), 0.02)
    assert ini.getboolean('test', 'fatal_errors')
    assert ini.get('runtime', 'sampler') == 'test'
    assert ini.get('output', 'filename') == 'chain.txt'

    with pytest.raises(ValueError):
        assert ini.getboolean('test', 'walkers')


@requires_cosmosis
def test_values(tx_config):
    values = _make_cosmosis_values(tx_config['parameters'])

    assert values.getfloat('params', 'Omega_k') == 0
    assert values.get('params', 'Omega_c').split() == ['0.25', '0.27', '0.32']

    with pytest.raises(cosmosis.runtime.config.CosmosisConfigurationError):
        assert values.getfloat('cosmological_parameters', 'Omega_k')


@requires_cosmosis
def test_pipeline(tx_config):
    data = None
    values = _make_cosmosis_values(tx_config['parameters'])
    pool = None
    pipeline = _make_cosmosis_pipeline(data, values, pool)

    # check all params made it through
    assert pipeline.nvaried == 2
    assert pipeline.nfixed == 8

    # check that parameter limits are right
    assert np.allclose(pipeline.min_vector(), np.array([0.25, 2.0e-9]))
    assert np.allclose(pipeline.max_vector(), np.array([0.32, 2.2e-9]))

    # check params have correct names
    assert pipeline.output_names() == ['params--omega_c', 'params--a_s']

    # Check the modules list is set up
    assert len(pipeline.modules) == 1
