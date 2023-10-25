import pytest
import numpy as np
from firecrown.models.cluster.mass_proxy import (
    MurataBinned,
    MurataUnbinned,
    MassRichnessGaussian,
)
from firecrown.models.cluster.kernel import (
    KernelType,
    Kernel,
    ArgReader,
)

PIVOT_Z = 0.6
PIVOT_MASS = 14.625862906


class MockArgsReader(ArgReader):
    def __init__(self):
        super().__init__()
        self.integral_bounds_idx = 0
        self.extra_args_idx = 1

    def get_independent_val(self, int_args, kernel_type: KernelType):
        bounds_values = int_args[self.integral_bounds_idx]
        return bounds_values[:, self.integral_bounds[kernel_type.name]]

    def get_extra_args(self, int_args, kernel_type: KernelType):
        extra_values = int_args[self.extra_args_idx]
        return extra_values[self.extra_args[kernel_type.name]]


@pytest.fixture(name="murata_binned_relation")
def fixture_murata_binned() -> MurataBinned:
    """Initialize cluster object."""

    mr = MurataBinned(PIVOT_MASS, PIVOT_Z)

    # Set the parameters to the values used in the test
    # they should be such that the variance is always positive.
    mr.mu_p0 = 3.00
    mr.mu_p1 = 0.086
    mr.mu_p2 = 0.01
    mr.sigma_p0 = 3.0
    mr.sigma_p1 = 0.07
    mr.sigma_p2 = 0.01

    return mr


@pytest.fixture(name="murata_unbinned_relation")
def fixture_murata_unbinned() -> MurataUnbinned:
    """Initialize cluster object."""

    mr = MurataUnbinned(PIVOT_MASS, PIVOT_Z)

    # Set the parameters to the values used in the test
    # they should be such that the variance is always positive.
    mr.mu_p0 = 3.00
    mr.mu_p1 = 0.086
    mr.mu_p2 = 0.01
    mr.sigma_p0 = 3.0
    mr.sigma_p1 = 0.07
    mr.sigma_p2 = 0.01

    return mr


def test_create_musigma_kernel():
    mb = MurataBinned(1, 1)
    assert isinstance(mb, Kernel)
    assert mb.kernel_type == KernelType.mass_proxy
    assert mb.is_dirac_delta is False
    assert mb.integral_bounds is None
    assert mb.has_analytic_sln is True
    assert mb.pivot_mass == 1 * np.log(10)
    assert mb.pivot_redshift == 1
    assert mb.log1p_pivot_redshift == np.log1p(1)

    assert mb.mu_p0 is None
    assert mb.mu_p1 is None
    assert mb.mu_p2 is None
    assert mb.sigma_p0 is None
    assert mb.sigma_p1 is None
    assert mb.sigma_p2 is None


def test_cluster_observed_z():
    for z in np.geomspace(1.0e-18, 2.0, 20):
        f_z = MassRichnessGaussian.observed_value((0.0, 0.0, 1.0), 0.0, z, 0, 0)
        assert f_z == pytest.approx(np.log1p(z), 1.0e-7, 0.0)


def test_cluster_observed_mass():
    for logM in np.linspace(10.0, 16.0, 20):
        f_logM = MassRichnessGaussian.observed_value((0.0, 1.0, 0.0), logM, 0.0, 0, 0)

        assert f_logM == pytest.approx(logM * np.log(10.0), 1.0e-7, 0.0)


def test_cluster_murata_binned_distribution(murata_binned_relation: MurataBinned):
    logM_array = np.linspace(7.0, 26.0, 20)
    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for logM_0, logM_1 in zip(logM_array[:-1], logM_array[1:]):
            extra_args = [(1.0, 5.0)]

            args1 = [np.array([[logM_0, z]]), extra_args]
            args2 = [np.array([[logM_1, z]]), extra_args]

            args_map = MockArgsReader()
            args_map.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}
            args_map.extra_args = {KernelType.mass_proxy.name: 0}

            probability_0 = murata_binned_relation.distribution(args1, args_map)
            probability_1 = murata_binned_relation.distribution(args2, args_map)

            assert probability_0 >= 0
            assert probability_1 >= 0

            # Probability should be initially monotonically increasing
            # and then monotonically decreasing. It should flip only once.

            # Test for the flip
            if (not flip) and (probability_1 < probability_0):
                flip = True

            # Test for the second flip
            if flip and (probability_1 > probability_0):
                raise ValueError("Probability flipped twice")

            if flip:
                assert probability_1 <= probability_0
            else:
                assert probability_1 >= probability_0


def test_cluster_murata_binned_mean(murata_binned_relation: MurataBinned):
    for mass in np.linspace(7.0, 26.0, 20):
        for z in np.geomspace(1.0e-18, 2.0, 20):
            test = murata_binned_relation.get_proxy_mean(mass, z)

            true = MassRichnessGaussian.observed_value(
                (3.00, 0.086, 0.01),
                mass,
                z,
                PIVOT_MASS * np.log(10.0),
                np.log1p(PIVOT_Z),
            )

            assert test == pytest.approx(true, rel=1e-7, abs=0.0)


def test_cluster_murata_binned_variance(murata_binned_relation: MurataBinned):
    for mass in np.linspace(7.0, 26.0, 20):
        for z in np.geomspace(1.0e-18, 2.0, 20):
            test = murata_binned_relation.get_proxy_sigma(mass, z)

            true = MassRichnessGaussian.observed_value(
                (3.00, 0.07, 0.01),
                mass,
                z,
                PIVOT_MASS * np.log(10.0),
                np.log1p(PIVOT_Z),
            )

            assert test == pytest.approx(true, rel=1e-7, abs=0.0)


def test_cluster_murata_unbinned_distribution(murata_unbinned_relation: MurataUnbinned):
    logM_array = np.linspace(7.0, 26.0, 20)
    for z in np.geomspace(1.0e-18, 2.0, 20):
        flip = False
        for logM_0, logM_1 in zip(logM_array[:-1], logM_array[1:]):
            extra_args = [2.5]

            args1 = [np.array([[logM_0, z]]), extra_args]
            args2 = [np.array([[logM_1, z]]), extra_args]

            args_map = MockArgsReader()
            args_map.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}
            args_map.extra_args = {KernelType.mass_proxy.name: 0}

            probability_0 = murata_unbinned_relation.distribution(args1, args_map)
            probability_1 = murata_unbinned_relation.distribution(args2, args_map)

            # Probability density should be initially monotonically increasing
            # and then monotonically decreasing. It should flip only once.

            # Test for the flip
            if (not flip) and (probability_1 < probability_0):
                flip = True

            # Test for the second flip
            if flip and (probability_1 > probability_0):
                raise ValueError("Probability flipped twice")

            if flip:
                assert probability_1 <= probability_0
            else:
                assert probability_1 >= probability_0
