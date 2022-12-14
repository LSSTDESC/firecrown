#!/usr/bin/env python
import os
import sys
import time

if sys.version_info[0] >= 3:
    unicode = str
import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
import firecrown.likelihood.gauss_family.statistic.source.number_counts as nc
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap

import sacc
import pyccl

# Load sacc file
saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_DIR}/examples/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits"
    )
)
sacc_data = sacc.Sacc.load_fits(saccfile)


# Define sources
n_source = 1
n_lens = 1
sources = {}
PTC = pyccl.nl_pt.PTCalculator(with_NC=True,
                             with_IA=True,
                             with_dd=False,
                             log10k_min=-4, log10k_max=2, nk_per_decade=20)
# Define the intrinsic alignment systematic. This will be added to the
# lensing sources later
#ia_systematic = wl.LinearAlignmentSystematic()

ia_systematic = wl.TATTSystematic()

for i in range(n_source):
    # Define the photo-z shift systematic.
    pzshift = wl.PhotoZShift(sacc_tracer=f"src{i}")
    mult_bias = wl.MultiplicativeShearBias(sacc_tracer=f"src{i}")

    # Create the weak lensing source, specifying the name of the tracer in the
    # sacc file and a list of systematics
    sources[f"src{i}"] = wl.WeakLensingPT(
        sacc_tracer=f"src{i}",
        systematics=[ia_systematic, pzshift], scale=2.0
    )

for i in range(n_lens):
    pzshift = nc.PhotoZShift(sacc_tracer=f"lens{i}")
    magnification = nc.MagnificationBiasSystematic(sacc_tracer=f"lens{i}")

    nl_bias = nc.NLBiasSystematic(sacc_tracer=f"lens{i}")
    sources[f"lens{i}"] = nc.NumberCountsPT(
        sacc_tracer=f"lens{i}",
        systematics=[pzshift, nl_bias], has_mag_bias=True, scale=3.0
    )

# Define the statistics we like to include in the likelihood
stats = {}
for stat, sacc_stat in [
    ("xip", "galaxy_shear_xi_plus"),
    ("xim", "galaxy_shear_xi_minus"),
]:
    for i in range(n_source):
        for j in range(i, n_source):
            # Define two-point statistics, given two sources (from above) and
            # the type of statistic.
            stats[f"{stat}_src{i}_src{j}"] = TwoPoint(
                source0=sources[f"src{i}"],
                source1=sources[f"src{j}"],
                sacc_data_type=sacc_stat, pt_calc = PTC
            )
for j in range(n_source):
    for i in range(n_lens):
        stats[f"gammat_lens{j}_src{i}"] = TwoPoint(
            source0=sources[f"lens{j}"],
            source1=sources[f"src{i}"],
            sacc_data_type="galaxy_shearDensity_xi_t",pt_calc=PTC
        )

for i in range(n_lens):
    stats[f"wtheta_lens{i}_lens{i}"] = TwoPoint(
        source0=sources[f"lens{i}"],
        source1=sources[f"lens{i}"],
        sacc_data_type="galaxy_density_xi", pt_calc=PTC
    )

# Create the likelihood from the statistics
lk = ConstGaussian(statistics=list(stats.values()))

# Read the two-point data from the sacc file
lk.read(sacc_data)

# To allow this likelihood to be used in cobaya or cosmosis, define a 
# an object called "likelihood" must be defined
likelihood = lk


# We can also run the likelihood directly
if __name__ == "__main__":
    import numpy as np
    import pyccl as ccl
    import matplotlib.pyplot as plt
    import pyccl.nl_pt as pt

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()

    # Set the parameters for our systematics
    systematics_params = ParamsMap({"ia_bias": 1.0,
                                    "alphaz": 0.0,
                                    "alphag": 0,
                                    "z_piv": 0.62,
                                    "ia_bias_ta" : 0.5,
                                    "ia_bias_2": 0.5,
                                    "alphaz_2":0.0, 
                                    "alphag_2":0.0,
                                    "src0_mult_bias": 0.0,
                                    
                                    "lens0_mag_bias":1.0,
                                    "lens0_bias":2.0,
                                    "lens0_bias_2":1.0,
                                    "lens0_bias_s":1.0,
                                    "lens0_delta_z":0.0,
                                    "lens0_alphaz":0.0,
                                    "lens0_alphag":0.0,
                                    "lens0_z_piv":0.62,
                                    "lens0_delta_z":0.000,


                                    "src0_delta_z": 0.000,
                                    "src1_delta_z": 0.000,
                                    "src2_delta_z": -0.001,
                                    "src3_delta_z": 0.002})

    # Apply the systematics parameters
    likelihood.update(systematics_params)
    # Compute the log-likelihood, using the ccl.Cosmology object as the input
    for i in range(2):
        start = time.time()
        log_like = likelihood.compute_loglike(ccl_cosmo)
        end = time.time()
        print(f"Log-like = {log_like:.1f}")
        print(f"time to compute = {end-start: .3f}")

        # Plot the predicted and measured statistic
        x = likelihood.statistics[0].ell_or_theta_
        y_data = likelihood.statistics[0].measured_statistic_
        y_err = np.sqrt(np.diag(likelihood.cov))[:len(x)]
        y_theory = likelihood.statistics[0].predicted_statistic_

        start = time.time()

        src0_tracer = sacc_data.get_tracer("src0")
        lens0_tracer = sacc_data.get_tracer("lens0")
        z, nz = src0_tracer.z, src0_tracer.nz
        lens_z, lens_nz = lens0_tracer.z, lens0_tracer.nz

        # Define a ccl.Cosmology object using default parameters
        #ccl_cosmo = ccl.CosmologyVanillaLCDM()
        #ccl_cosmo.compute_nonlin_power()

        # Bare CCL setup
        a_1 = 1.
        a_2 = 0.5
        a_d = 0.5

        b_1 = 2.0
        b_2 = 1.0
        b_s = 1.0

        mag_bias = 1.0

        c_1, c_d, c_2 = pt.translate_IA_norm(ccl_cosmo, z, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2=False)

        #print(c_d)

        # Code that creates a Pk2D object:
        ptc = pt.PTCalculator(with_NC=True, with_IA=True,
                              log10k_min=-4, log10k_max=2, nk_per_decade=20)
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(z, c_1), c2=(z, c_2), cdelta=(z, c_d))
        ptt_m = pt.PTMatterTracer()
        ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s)
        # IA
        
        pk_im = pt.get_pt_pk2d(ccl_cosmo, ptt_i, tracer2=ptt_m, ptc=ptc)
        pk_ii = pt.get_pt_pk2d(ccl_cosmo, ptt_i, ptc=ptc)
        pk_gi = pt.get_pt_pk2d(ccl_cosmo, ptt_g, tracer2=ptt_i, ptc=ptc)
        # Galaxies
        pk_gm = pt.get_pt_pk2d(ccl_cosmo, ptt_g, tracer2=ptt_m, ptc=ptc)
        pk_gg = pt.get_pt_pk2d(ccl_cosmo, ptt_g, ptc=ptc)
        # Magnification

        # Plot the predicted and measured statistic
        x = likelihood.statistics[0].ell_or_theta_
        x_minus = likelihood.statistics[1].ell_or_theta_
        x_ggl = likelihood.statistics[2].ell_or_theta_
        x_nc = likelihood.statistics[3].ell_or_theta_
        y_data = likelihood.statistics[0].measured_statistic_
        y_err = np.sqrt(np.diag(likelihood.cov))[:len(x)]
        y_theory = likelihood.statistics[0].predicted_statistic_
        y_theory_minus = likelihood.statistics[1].predicted_statistic_
        y_theory_ggl = likelihood.statistics[2].predicted_statistic_
        y_theory_nc = likelihood.statistics[3].predicted_statistic_

        #print(list(likelihood.statistics[0].cells.keys()))

        ells = likelihood.statistics[0].ells
        ells_minus = likelihood.statistics[1].ells
        ells_ggl = likelihood.statistics[2].ells
        ells_nc = likelihood.statistics[3].ells

        # Code that computes effect from IA using that Pk2D object
        t_lens = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz))
        t_ia = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz), has_shear=False, ia_bias=(z, np.ones_like(z)), use_A_ia=False)
        t_g = ccl.NumberCountsTracer(ccl_cosmo, has_rsd=False, dndz=(lens_z, lens_nz), bias=(lens_z, np.ones_like(lens_z)))
        t_m = ccl.NumberCountsTracer(ccl_cosmo, has_rsd=False, dndz=(lens_z, lens_nz), bias=None, mag_bias=(lens_z, np.ones_like(lens_z)*mag_bias))
        cl_GI = ccl.angular_cl(ccl_cosmo, t_lens, t_ia, ells, p_of_k_a=pk_im)*4.0
        cl_II = ccl.angular_cl(ccl_cosmo, t_ia, t_ia, ells, p_of_k_a=pk_ii)*4.0
        # The weak gravitational lensing power spectrum
        cl_GG = ccl.angular_cl(ccl_cosmo, t_lens, t_lens, ells)*4.0

        # Galaxies
        cl_gG = ccl.angular_cl(ccl_cosmo, t_g, t_lens, ells_ggl, p_of_k_a=pk_gm) * 6.0
        cl_gI = ccl.angular_cl(ccl_cosmo, t_g, t_ia, ells_ggl, p_of_k_a=pk_gi) * 6.0
        cl_gg = ccl.angular_cl(ccl_cosmo, t_g, t_g, ells_nc, p_of_k_a=pk_gg) * 9.0
        # Magnification
        cl_mI = ccl.angular_cl(ccl_cosmo, t_m, t_ia, ells_ggl, p_of_k_a=pk_im)* 6.0
        cl_gm = ccl.angular_cl(ccl_cosmo, t_g, t_m, ells_nc, p_of_k_a=pk_gm) * 9.0
        cl_mm = ccl.angular_cl(ccl_cosmo, t_m, t_m, ells_nc) * 9.0
        cl_mG = ccl.angular_cl(ccl_cosmo, t_m, t_lens, ells_ggl) * 6.0

        # The observed angular power spectrum is the sum of the two.
        cl_cs_theory = cl_GG + 2*cl_GI + cl_II  # normally we would also have a third term, +cl_II).
        #print(cl_GG, cl_GI, cl_II)

        ang = ccl.correlation(ccl_cosmo, ells, cl_cs_theory, likelihood.statistics[0].ell_or_theta_/60, type="GG+")
        ang_minus = ccl.correlation(ccl_cosmo, ells_minus, cl_cs_theory, likelihood.statistics[1].ell_or_theta_/60, type="GG-")
       
        ang_ggl = ccl.correlation(ccl_cosmo, ells_ggl, cl_gI+cl_gG+cl_mI + cl_mG, likelihood.statistics[2].ell_or_theta_/60, type="NG")
        ang_nc = ccl.correlation(ccl_cosmo, ells_nc, cl_gg+2*cl_gm+cl_mm, likelihood.statistics[3].ell_or_theta_/60, type="NN")
        end = time.time()
        print(f"time to compute = {end-start: .3f}")

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    fig.subplots_adjust(hspace=0)
    # ax[0].plot(x, y_theory, label="Total")
    #ax[0].plot(ells, cells_GG, label="GG firecrown")
    #ax[0].plot(ells, cl_GG, ls="--", label="GG CCL")
    #ax[0].plot(ells, -cells_GI, label="-GI firecrown")
    #ax[0].plot(ells, -cl_GI, ls="--", label="-GI CCL")
    #ax[0].plot(ells, cells_II, label="II firecrown")
    #ax[0].plot(ells, cl_II, ls="--", label="II CCL")
    #ax[0].plot(ells, cells_cs_total, label="total CS firecrown")
    #ax[0].plot(ells, cl_cs_theory, ls="--", label="total CS CCL")

    ax[0].plot(x, y_theory, label="firecrown xi_+")
    ax[0].plot(x_minus, y_theory_minus, label="firecrown xi_-")
    ax[0].plot(x_ggl, y_theory_ggl, label="firecrown ggl")
    ax[0].plot(x_nc, y_theory_nc, label="firecrown nc")
    ax[0].plot(x, ang, label="CCL xi_+", linestyle='--')
    ax[0].plot(x_minus, ang_minus, label="CCL xi_+", linestyle='--')
    ax[0].plot(x_ggl, ang_ggl, label="CCL ggl", linestyle='--')
    ax[0].plot(x_nc, ang_nc, label="CCL nc", linestyle='--')

    #ax[0].errorbar(x, y_data, y_err, ls="none", marker="o")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\theta$ [arcmin]")
    ax[0].set_ylabel(r"$\xi(\theta)$")
    ax[0].legend()

    ax[1].plot(x, (y_theory-ang)/ang, label="xi_+ residual")
    ax[1].plot(x_minus, (y_theory_minus-ang_minus)/ang_minus, label="xi_- residual")
    ax[1].plot(x_ggl, (y_theory_ggl-ang_ggl)/ang_ggl, label="ggl residual")
    ax[1].plot(x_nc, (y_theory_nc-ang_nc)/ang_nc, label="nc residual")

    ax[1].set_xlabel(r"$\theta$ [arcmin]")
    ax[1].set_ylabel(r"$\xi_+(\theta)$ residual")
    ax[1].set_yscale("linear")
    ax[1].set_xscale("log")
    ax[1].legend()
    #plt.show()




    '''
    ax[1].plot(ells, -cells_gI, label="-Ig firecrown")
    ax[1].plot(ells, -cl_gI, ls="--", label="-Ig CCL")
    ax[1].plot(ells, cells_gG, label="Gg firecrown")
    ax[1].plot(ells, cl_gG, ls="--", label="Gg CCL")
    ax[1].plot(ells, cells_gg, label="gg firecrown")
    ax[1].plot(ells, cl_gg, ls="--", label="gg CCL")
    ax[1].plot(ells, -cells_mI, label="-mI firecrown")
    ax[1].plot(ells, -cl_mI, ls="--", label="-mI CCL")
    ax[1].plot(ells, cells_gm, label="gm firecrown")
    ax[1].plot(ells, cl_gm, ls="--", label="gm CCL")
    '''
    # ax[0].errorbar(x, y_data, y_err, ls="none", marker="o")
    ax[0].set_xscale("log")
    #[a.set_yscale("log") for a in ax]
    #ax[1].set_xlabel(r"$\ell$")
    #[a.set_ylabel(r"$C_\ell$") for a in ax]
    #ax[1].set_ylabel(r"$C_\ell$")
    [a.legend(fontsize="small") for a in ax]

    fig.suptitle("PT Cls, including IA, galaxy bias, magnification")
    fig.savefig("plots/pt_cls.png", facecolor="white", dpi=300)

    plt.show()