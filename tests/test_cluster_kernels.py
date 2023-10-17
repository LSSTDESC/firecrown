import pytest
import numpy as np
from firecrown.models.cluster.kernel import (
    Completeness,
    Purity,
    KernelType,
    Kernel,
    MassRichnessMuSigma,
    DESY1PhotometricRedshift,
    SpectroscopicRedshift,
    TrueMass,
    ArgsMapping,
)


def test_create_desy1_photometric_redshift_kernel():
    drk = DESY1PhotometricRedshift()
    assert isinstance(drk, Kernel)
    assert drk.kernel_type == KernelType.z_proxy
    assert drk.is_dirac_delta is False
    assert drk.integral_bounds is None
    assert drk.has_analytic_sln is False


def test_create_spectroscopic_redshift_kernel():
    srk = SpectroscopicRedshift()
    assert isinstance(srk, Kernel)
    assert srk.kernel_type == KernelType.z_proxy
    assert srk.is_dirac_delta is True
    assert srk.integral_bounds is None
    assert srk.has_analytic_sln is False


def test_create_musigma_kernel():
    msk = MassRichnessMuSigma(1, 1)
    assert isinstance(msk, Kernel)
    assert msk.kernel_type == KernelType.mass_proxy
    assert msk.is_dirac_delta is False
    assert msk.integral_bounds is None
    assert msk.has_analytic_sln is True


def test_create_mass_kernel():
    mk = TrueMass()
    assert isinstance(mk, Kernel)
    assert mk.kernel_type == KernelType.mass_proxy
    assert mk.is_dirac_delta is True
    assert mk.integral_bounds is None
    assert mk.has_analytic_sln is False


def test_create_completeness_kernel():
    ck = Completeness()
    assert isinstance(ck, Kernel)
    assert ck.kernel_type == KernelType.completeness
    assert ck.is_dirac_delta is False
    assert ck.integral_bounds is None
    assert ck.has_analytic_sln is False


def test_create_purity_kernel():
    pk = Purity()
    assert isinstance(pk, Kernel)
    assert pk.kernel_type == KernelType.purity
    assert pk.is_dirac_delta is False
    assert pk.integral_bounds is None
    assert pk.has_analytic_sln is False


def test_spec_z_distribution():
    srk = SpectroscopicRedshift()
    assert srk.distribution([0.5], ArgsMapping()) == 1.0


def test_true_mass_distribution():
    tmk = TrueMass()
    assert tmk.distribution([0.5], ArgsMapping()) == 1.0


def test_purity_distribution():
    pk = Purity()
    mass_proxy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    arguments = np.array(list(zip(mass_proxy, z)))
    map = ArgsMapping()
    map.integral_bounds = {KernelType.mass_proxy.name: 0, KernelType.z.name: 1}

    truth = np.array(
        [
            0.77657274,
            0.96966127,
            0.99286409,
            0.99780586,
            0.999224,
            0.99970302,
            0.99988111,
            0.99995125,
            0.99997982,
            0.99999166,
        ]
    )

    purity = pk.distribution([arguments], map)
    for ref, true in zip(purity, truth):
        assert ref == pytest.approx(true, rel=1e-7, abs=0.0)


def test_completeness_distribution():
    ck = Completeness()
    mass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    bounds = np.array(list(zip(mass, z)))
    map = ArgsMapping()
    map.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}

    truth = np.array(
        [
            0.0056502277493542,
            0.01896566878380423,
            0.03805597500308377,
            0.06224888967250564,
            0.09124569979282898,
            0.12486247682690908,
            0.16290218589569144,
            0.20507815091349266,
            0.2509673905442634,
            0.2999886170051561,
        ]
    )

    comp = ck.distribution([bounds], map)
    for ref, true in zip(comp, truth):
        assert ref == pytest.approx(true, rel=1e-7, abs=0.0)


def test_des_photoz_kernel_distribution():
    dpk = DESY1PhotometricRedshift()

    mass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    z_proxy = np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01])

    bounds = np.array(list(zip(mass, z, z_proxy)))

    map = ArgsMapping()
    map.integral_bounds = {
        KernelType.mass.name: 0,
        KernelType.z.name: 1,
        KernelType.z_proxy.name: 2,
    }

    truth = [
        7.134588921656481,
        6.557328601698999,
        6.065367634804254,
        5.641316284718016,
        5.272157878477569,
        4.9479710868093685,
        4.661070179674804,
        4.405413986167644,
        4.176191421334415,
        3.969525474770118,
    ]

    spread = dpk.distribution([bounds], map)
    for ref, true in zip(spread, truth):
        assert ref == pytest.approx(true, rel=1e-7, abs=0.0)


# class ArgsMapping:
#     def __init__(self):
#         self.integral_bounds = dict()
#         self.extra_args = dict()

#         self.integral_bounds_idx = 0
#         self.extra_args_idx = 1

#     def get_integral_bounds(self, int_args, kernel_type: KernelType):
#         bounds_values = int_args[self.integral_bounds_idx]
#         return bounds_values[:, self.integral_bounds[kernel_type.name]]

#     def get_extra_args(self, int_args, kernel_type: KernelType):
#         extra_values = int_args[self.extra_args_idx]
#         return extra_values[self.extra_args[kernel_type.name]]


def test_args_mapper_extra_args():
    args_map = ArgsMapping()

    mass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    args_map.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}

    extra_args = [(1, 2, 3), ("hello world")]
    args_map.extra_args = {KernelType.mass_proxy.name: 0, KernelType.z_proxy.name: 1}

    integral_bounds = np.array(list(zip(mass, z)))
    int_args = [integral_bounds, extra_args]

    assert args_map.get_extra_args(int_args, KernelType.mass_proxy) == (1, 2, 3)
    assert args_map.get_extra_args(int_args, KernelType.z_proxy) == "hello world"


def test_args_mapper_integral_bounds():
    args_map = ArgsMapping()

    mass = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    z_proxy = np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01])
    mass_proxy = np.array([0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9])

    args_map.integral_bounds = {
        KernelType.mass.name: 0,
        KernelType.z.name: 1,
        KernelType.z_proxy.name: 2,
        KernelType.mass_proxy.name: 3,
    }

    integral_bounds = np.array(list(zip(mass, z, z_proxy, mass_proxy)))
    int_args = [integral_bounds]

    assert (mass == args_map.get_integral_bounds(int_args, KernelType.mass)).all()
    assert (z == args_map.get_integral_bounds(int_args, KernelType.z)).all()
    assert (z_proxy == args_map.get_integral_bounds(int_args, KernelType.z_proxy)).all()
    assert (
        mass_proxy == args_map.get_integral_bounds(int_args, KernelType.mass_proxy)
    ).all()


def test_create_args_mapping():
    am = ArgsMapping()
    assert am.integral_bounds == dict()
    assert am.extra_args == dict()
    assert am.integral_bounds_idx == 0
    assert am.extra_args_idx == 1


# int_args = [
#     (values im integrating over),
#     (extra arguments for my integrand),
#     args_mapper = {
#         integral_bounds = {
#             mass at index 0
#             z at index 1
#             z_proxy at index 2
#         }
#         extra_args = {
#             mass_proxy at index 0
#         }
#     }
# ]

# models/Updatable
#   firecrown/
#   cluster_abundance/
# cluster_abundance/
#   firecrown/
