"""Some utility functions for patterns common in Firecrown."""

from typing import Generator, TypeVar, Type, Callable

import functools
import numpy as np
import pyccl
import scipy.interpolate
from numpy import typing as npt
from pydantic import BaseModel

import sacc

import yaml
from yaml import CLoader as Loader
from yaml import CDumper as Dumper

ST = TypeVar("ST")  # This will be used in YAMLSerializable


class YAMLSerializable:
    """Protocol for classes that can be serialized to and from YAML."""

    def to_yaml(self: ST) -> str:
        """Return the YAML representation of the object."""
        return yaml.dump(self, Dumper=Dumper, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[ST], yaml_str: str) -> ST:
        """Load the object from YAML."""
        return yaml.load(yaml_str, Loader=Loader)


def base_model_from_yaml(cls: type, yaml_str: str):
    """Create a base model from a yaml string."""
    if not issubclass(cls, BaseModel):
        raise ValueError("cls must be a subclass of pydantic.BaseModel")

    try:
        return cls.model_validate(
            yaml.safe_load(yaml_str),
            strict=True,
        )
    except Exception as e:
        raise ValueError(
            f"Error creating {cls.__name__} from yaml. Parsing error message:\n{e}"
        ) from e


def base_model_to_yaml(model: BaseModel) -> str:
    """Convert a base model to a yaml string."""
    return yaml.dump(model.model_dump(), default_flow_style=None, sort_keys=False)


def upper_triangle_indices(n: int) -> Generator[tuple[int, int], None, None]:
    """Returns the upper triangular indices for an (n x n) matrix.

    generator that yields a sequence of tuples that carry the indices for an
    (n x n) upper-triangular matrix. This is a replacement for the nested loops:

    for i in range(n):
      for j in range(i, n):
        ...

    :param n: the size of the matrix
    :return: the generator
    """
    for i in range(n):
        for j in range(i, n):
            yield i, j


def save_to_sacc(
    sacc_data: sacc.Sacc,
    data_vector: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
    strict: bool = True,
) -> sacc.Sacc:
    """Save a data vector into a (new) SACC object, copied from `sacc_data`.

    Note that the original object `sacc_data` is not modified. Its contents are
    copied into a new object, and the new information is put into that copy,
    which is returned by this method.

    If `strict` is True (the default), then we must overwrite the entire data
    vector. If `strict` is False, then we only overwrite the data at the
    specified indices.

    :param sacc_data: SACC object to be copied. It is not modified.
    :param data_vector: Data vector to be saved to the new copy of `sacc_data`.
    :param indices: SACC indices where the data vector should be written.
    :param strict: Whether to check if the data vector covers all the data
        already present in the sacc_data.
    :return: A copy of `sacc_data`, with data at `indices` replaced with `data_vector`.
    """
    assert len(indices) == len(data_vector)

    new_sacc = sacc_data.copy()

    if strict:
        if set(indices.ravel().tolist()) != set(sacc_data.indices()):
            raise RuntimeError(
                "The data to be saved does not cover all the data in the "
                "sacc object. To write only the calculated predictions, "
                "set strict=False."
            )

    for data_idx, sacc_idx in enumerate(indices):
        new_sacc.data[sacc_idx].value = data_vector[data_idx]

    return new_sacc


def compare_optional_arrays(x: None | npt.NDArray, y: None | npt.NDArray) -> bool:
    """Compare two arrays, allowing for either or both to be None.

    :param x: first array
    :param y: second array
    :return: whether the arrays are equal
    """
    if x is None and y is None:
        return True
    if x is not None and y is not None:
        return np.array_equal(x, y)
    # One is None and the other is not.
    return False


def compare_optionals(x: None | object, y: None | object) -> bool:
    """Compare two objects, allowing for either or both to be None.

    :param x: first object
    :param y: second object
    :return: whether the objects are equal
    """
    if x is None and y is None:
        return True
    if x is not None and y is not None:
        return x == y
    # One is None and the other is not.
    return False


@functools.lru_cache(maxsize=128)
def cached_angular_cl(
    cosmo: pyccl.Cosmology,
    tracers: tuple[pyccl.Tracer, pyccl.Tracer],
    ells: npt.NDArray[np.int64],
    p_of_k_a=None | Callable[[npt.NDArray[np.int64]], npt.NDArray[np.float64]],
):
    """Wrapper for pyccl.angular_cl, with automatic caching.

    :param cosmo: the current cosmology
    :param tracers: tracers indicating the measurements to be correlated
    :param ells: ell values at which to calculate the power spectrum
    :param p_of_k_a: function that computes the power spectrum
    """
    return pyccl.angular_cl(
        cosmo, tracers[0], tracers[1], np.array(ells), p_of_k_a=p_of_k_a
    )


def make_log_interpolator(
    x: npt.NDArray[np.int64], y: npt.NDArray[np.float64]
) -> Callable[[npt.NDArray[np.int64]], npt.NDArray[np.float64]]:
    """Return a function object that does 1D spline interpolation.

    If all the y values are greater than 0, the function
    interpolates log(y) as a function of log(x).
    Otherwise, the function interpolates y as a function of log(x).
    The resulting interpolater will not extrapolate; if called with
    an out-of-range argument it will raise a ValueError.
    """
    if np.all(y > 0):
        # use log-log interpolation
        intp = scipy.interpolate.InterpolatedUnivariateSpline(
            np.log(x), np.log(y), ext=2
        )

        def log_log_interpolator(x_: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
            """Interpolate on log-log scale."""
            return np.exp(intp(np.log(x_)))

        return log_log_interpolator
    # only use log for x
    intp = scipy.interpolate.InterpolatedUnivariateSpline(np.log(x), y, ext=2)

    def log_x_interpolator(x_: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Interpolate on log-x scale."""
        return intp(np.log(x_))

    return log_x_interpolator
