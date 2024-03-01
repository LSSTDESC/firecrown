import numpy as np

import matplotlib.pyplot as plt

import sacc

import pyccl as ccl

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.parameters import ParamsMap

from firecrown.factory.factory import build_likelihood


config_str = """
two-point:
    weak_lensing:
        global_systematics:
            IA:
                class_path: firecrown.likelihood.gauss_family.statistic.source.weak_lensing.TattAlignmentSystematic

        per_bin_systematics:
            photoz_shift:
                class_path: firecrown.likelihood.gauss_family.statistic.source.weak_lensing.PhotoZShift

    number_counts:
        per_bin_systematics:
            galaxy_bias:
                class_path: firecrown.likelihood.gauss_family.statistic.source.number_counts.PTNonLinearBiasSystematic

modeling_tools:
    pt_calculator:
        class_path: pyccl.nl_pt.EulerianPTCalculator
        init_args:
            with_NC: True
            with_IA: True
            log10k_min: -4
            log10k_max: 2
            nk_per_decade: 20

likelihood:
    class_path: firecrown.likelihood.gauss_family.gaussian.ConstGaussian

data:
    # sacc: des_y1_3x2pt_sacc_data.fits
    source_tracer_name: "src"
    lens_tracer_name: "lens"
    statistics:
        galaxy_shear_xi_plus:
            # src0-src0:
            #     remove: True
            # src3-src3:
            #     ell_min: 12
            #     ell_max: 1000
        galaxy_shear_xi_minus:
"""


likelihood, tools = build_likelihood(
    dict(
        config=config_str
    ),
    sacc_data=sacc.Sacc.load_fits("des_y1_3x2pt_sacc_data.fits")
)

print("Likelihood parameters:", list(likelihood.required_parameters().get_params_names()))
print("Modelling tools parameters:", list(tools.required_parameters().get_params_names()))



ccl_cosmo = ccl.Cosmology(
    Omega_c =  0.2905-0.0473,
    h       =  0.6896,
    Omega_b =  0.0473,
    n_s     =  0.969,
    A_s     =  2.19e-9,
    w0      =  -1.0,
)

params = ParamsMap(
    {
        "ia_a_1": 1.0,
        "ia_a_2": 0.5,
        "ia_a_d": 0.5,
        "lens0_bias": 2.0,
        "lens0_b_2": 1.0,
        "lens0_b_s": 1.0,
        "lens1_bias": 2.0,
        "lens1_b_2": 1.0,
        "lens1_b_s": 1.0,
        "lens2_bias": 2.0,
        "lens2_b_2": 1.0,
        "lens2_b_s": 1.0,
        "lens3_bias": 2.0,
        "lens3_b_2": 1.0,
        "lens3_b_s": 1.0,
        "lens4_bias": 2.0,
        "lens4_b_2": 1.0,
        "lens4_b_s": 1.0,
        "lens0_mag_bias": 1.0,
        "lens1_mag_bias": 1.0,
        "lens2_mag_bias": 1.0,
        "lens3_mag_bias": 1.0,
        "lens4_mag_bias": 1.0,
        "src0_delta_z": 0.000,
        "src1_delta_z": 0.000,
        "src2_delta_z": 0.000,
        "src3_delta_z": 0.000,
        "lens0_delta_z": 0.000,
        "lens1_delta_z": 0.000,
        "lens2_delta_z": 0.000,
        "lens3_delta_z": 0.000,
        "lens4_delta_z": 0.000,
    }
)

likelihood.reset()
tools.reset()

likelihood.update(params)
tools.update(params)
tools.prepare(ccl_cosmo)

chi2 = likelihood.compute_chisq(tools)
print(f"{chi2=:.2f}")
