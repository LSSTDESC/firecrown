#!/usr/bin/env python

"""Example factory function for DES Y1 3x2pt likelihood."""
from dataclasses import dataclass
import os

import numpy as np
import sacc
import pyccl as ccl
import pyccl.nl_pt

import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood.two_point import (
    TwoPoint,
    TracerNames,
    TRACER_NAMES_TOTAL,
)
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.likelihood import Likelihood

# change this!!
saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_DIR}/examples/des_y1_3x2pt/paul_auto_forecast_fid_3x2pt_linear_sys_limber_20_log_bins.sacc"  # needs to change!
    )
)


def build_likelihood(_) -> tuple[Likelihood, ModelingTools]:
    """Likelihood factory function for DES Y1 3x2pt analysis."""
    # Load sacc file
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # Define the intrinsic alignment systematic. This will be added to the
    # lensing sources later
    ia_systematic = wl.TattAlignmentSystematic()

    num_lens = 5
    num_srcs = 5
    # lens, src order
    ggl_combos = [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4), (3, 4)]
    lens_tracers = []
    src_tracers = []

    for i in range(num_lens):
        name = "lens" + str(i)
        lens_pzshift = nc.PhotoZShift(sacc_tracer=name)
        magnification = nc.ConstantMagnificationBiasSystematic(sacc_tracer=name)
        nl_bias = nc.PTNonLinearBiasSystematic(sacc_tracer=name)
        lens_tracer = nc.NumberCounts(
            sacc_tracer=name,
            has_rsd=True,
            systematics=[lens_pzshift, magnification, nl_bias],
        )
        lens_tracers.append(lens_tracer)

    for i in range(num_srcs):
        name = "src" + str(i)
        src_pzshift = wl.PhotoZShift(sacc_tracer=name)
        # Create the weak lensing source, specifying the name of the tracer in the
        # sacc file and a list of systematics
        src_tracer = wl.WeakLensing(
            sacc_tracer=name, systematics=[src_pzshift, ia_systematic]
        )
        src_tracers.append(src_tracer)

    statistics = []
    for i in range(num_lens):
        statistics.append(
            TwoPoint(
                source0=lens_tracers[i],
                source1=lens_tracers[i],
                sacc_data_type="galaxy_density_cl",
            )
        )

    for lbin, sbin in ggl_combos:
        statistics.append(
            TwoPoint(
                source0=lens_tracers[lbin],
                source1=src_tracers[sbin],
                sacc_data_type="galaxy_shearDensity_cl_e",
            )
        )

    for i in range(num_srcs):
        for j in range(i, num_srcs):
            statistics.append(
                TwoPoint(
                    source0=src_tracers[i],
                    source1=src_tracers[j],
                    sacc_data_type="galaxy_shear_cl_ee",
                )
            )

    # Create the likelihood from the statistics
    pt_calculator = pyccl.nl_pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=20,
    )

    modeling_tools = ModelingTools(pt_calculator=pt_calculator)
    # Note that the ordering of the statistics is relevant, because the data
    # vector, theory vector, and covariance matrix will be organized to
    # follow the order used here.
    likelihood = ConstGaussian(statistics=statistics)

    # Read the two-point data from the sacc file
    likelihood.read(sacc_data)

    # an object called "likelihood" must be defined
    print(
        "Using parameters:", list(likelihood.required_parameters().get_params_names())
    )

    """
    systematics_params = ParamsMap(
        {
            "ia_a_1": cs.a_1,
            "ia_a_2": cs.a_2,
            "ia_a_d": cs.a_d,
            "lens0_bias": cs.b_1,
            "lens0_b_2": cs.b_2,
            "lens0_b_s": cs.b_s,
            "lens0_mag_bias": cs.mag_bias,
            "src0_delta_z": 0.000,
            "lens0_delta_z": 0.000,
        }
    )
    """
    # for default/initial values?
    # likelihood.update(systematics_params)
    # tools.update(systematics_params)

    # To allow this likelihood to be used in cobaya or cosmosis,
    # return the likelihood object
    return likelihood, modeling_tools
