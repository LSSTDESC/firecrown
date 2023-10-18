import numpy as np
from firecrown.models.cluster.kernel import (
    KernelType,
    ArgsMapping,
)


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
