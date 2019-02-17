import pytest
from ..run import _make_parallel_pool, _make_cosmosis_params
from ..run import _make_cosmosis_values, _make_cosmosis_pipeline
from ..run import run_cosmosis
import yaml
import numpy as np
import os
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
  # Parameters varied with cosmosis
  # need a min value, starting point, and max value,
  # like so:
  Omega_c: [0.25, 0.27, 0.32]
  Omega_b: 0.045
  h: 0.67
  n_s: 0.96
  A_s: [2.0e-9, 2.1e-9, 2.2e-9]
  w0: -1.0
  wa: 0.0

  # lens bin zero
  src0_delta_z: [-0.1, 0.0, 0.1]
  src1_delta_z: 0.0


cosmosis:
  sampler: test
  output: chain.txt
  debug: True
  quiet: True
  mpi: False
  # parameters for individual samplers:
  test:
    # pretending this sampler can take all these options
    fatal_errors: True
    walkers: 10
    nsample: 20
    nsample_dimension: 5
    step_size: 0.02
  grid:
    nsample_dimension: 5
"""
    config = yaml.load(config_text)
    return config


@requires_cosmosis
def test_pool(tx_config):
    pool = _make_parallel_pool(tx_config['cosmosis'])
    assert pool is None or pool.size > 0
    assert pool is None or isinstance(pool, cosmosis.runtime.mpi_pool.MPIPool)


@requires_cosmosis
def test_config(tx_config):
    ini = _make_cosmosis_params(tx_config['cosmosis'])
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
    assert pipeline.nvaried == 3
    assert pipeline.nfixed == 7

    # check that parameter limits are right
    assert np.allclose(pipeline.min_vector(), np.array([0.25, 2.0e-9, -0.1]))
    assert np.allclose(pipeline.max_vector(), np.array([0.32, 2.2e-9, 0.1]))

    # check params have correct names
    assert pipeline.output_names() == [
        'params--omega_c', 'params--a_s', 'params--src0_delta_z']

    # Check the modules list is set up
    assert len(pipeline.modules) == 1


@requires_cosmosis
def test_sampling(tx_config, tmpdir):
    chain_file = os.path.join(tmpdir, 'chain.txt')
    tx_config['cosmosis']['sampler'] = 'grid'
    tx_config['cosmosis']['output'] = chain_file
    # Run with no likelihoods, so that posterior = constant
    run_cosmosis(tx_config, tx_config)
    chain = np.loadtxt(chain_file)
    # Check the grid sampler has made the right number of points
    assert len(chain) == 125
    # Check the grid points are in the right place
    assert np.allclose(np.unique(chain[:, 0]), np.linspace(0.25, 0.32, 5))
    assert np.allclose(np.unique(chain[:, 1]), np.linspace(2.0e-9, 2.1e-9, 5))
    assert np.allclose(np.unique(chain[:, 2]), np.linspace(-0.1, 0.1, 5))
    # Check the posteriors are all the same.
    # They are not zero because of the prior.
    assert np.allclose(np.unique(chain[:, 3]), chain[0, 3])
