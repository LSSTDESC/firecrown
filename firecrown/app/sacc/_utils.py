"""Utility functions for SACC data analysis."""

from typing import TypedDict

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from firecrown import metadata_types as mdt


QuadOpts = TypedDict(
    "QuadOpts",
    {
        "limit": int,
        "epsabs": float,
        "epsrel": float,
    },
)


def mean_std_tracer(tracer: mdt.InferredGalaxyZDist):
    """Compute the mean and standard deviation of a tracer.

    :param tracer: The galaxy redshift distribution tracer to analyze.
    :return: Tuple of (mean_z, std_z) for the tracer distribution.
    """
    # Create monotonic spline
    spline = PchipInterpolator(tracer.z, tracer.dndz, extrapolate=False)
    quad_opts: QuadOpts = {"limit": 10000, "epsabs": 0.0, "epsrel": 1.0e-3}

    def spline_func(t):
        return spline(t)

    # Normalization
    norm, _ = quad(spline_func, tracer.z[0], tracer.z[-1], **quad_opts)

    # Mean
    mean_z, _ = quad(lambda t: t * spline(t), tracer.z[0], tracer.z[-1], **quad_opts)
    mean_z /= norm

    # Variance
    var_z, _ = quad(
        lambda t: (t - mean_z) ** 2 * spline(t), tracer.z[0], tracer.z[-1], **quad_opts
    )
    std_z = np.sqrt(var_z / norm)

    return mean_z, std_z
