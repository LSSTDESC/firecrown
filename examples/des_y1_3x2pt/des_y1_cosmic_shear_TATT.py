#!/usr/bin/env python
import os

import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.likelihood import PTSystematic
from firecrown.parameters import ParamsMap

import sacc


# Load sacc file
saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_DIR}/examples/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits"
    )
)
sacc_data = sacc.Sacc.load_fits(saccfile)


# Define sources
n_source = 1
sources = {}

# Define the intrinsic alignment systematic. This will be added to the
# lensing sources later
ia_systematic = wl.TattAlignmentSystematic()

for i in range(n_source):
    # Define the photo-z shift systematic.
    pzshift = wl.PhotoZShift(sacc_tracer=f"src{i}")

    # Create the weak lensing source, specifying the name of the tracer in the
    # sacc file and a list of systematics
    sources[f"src{i}"] = wl.WeakLensing(
        sacc_tracer=f"src{i}", systematics=[pzshift, ia_systematic]
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
                sacc_data_type=sacc_stat,
            )

# Create the likelihood from the statistics
pt_systematic = PTSystematic(
    with_NC=False,
    with_IA=True,
    with_dd=True,
    log10k_min=-4,
    log10k_max=2,
    nk_per_decade=20,
)
lk = ConstGaussian(statistics=list(stats.values()), systematics=[pt_systematic])

# Read the two-point data from the sacc file
lk.read(sacc_data)

# To allow this likelihood to be used in cobaya or cosmosis, define a
# an object called "likelihood" must be defined
likelihood = lk
print("Using parameters:", list(lk.required_parameters().get_params_names()))


# We can also run the likelihood directly
if __name__ == "__main__":
    import numpy as np
    import pyccl as ccl
    import pyccl.nl_pt as pt
    import matplotlib.pyplot as plt

    src0_tracer = sacc_data.get_tracer("src0")
    z, nz = src0_tracer.z, src0_tracer.nz

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()
    ccl_cosmo.compute_nonlin_power()

    # Bare CCL setup
    a_1 = 1.0
    a_2 = 0.5
    a_d = 0.5
    c_1, c_d, c_2 = pt.translate_IA_norm(
        ccl_cosmo, z, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2=False
    )

    # Code that creates a Pk2D object:
    ptc = pt.PTCalculator(
        with_NC=True, with_IA=True, log10k_min=-4, log10k_max=2, nk_per_decade=20
    )
    ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(z, c_1), c2=(z, c_2), cdelta=(z, c_d))
    ptt_m = pt.PTMatterTracer()
    # IAs x matter
    pk_im = pt.get_pt_pk2d(ccl_cosmo, ptt_i, tracer2=ptt_m, ptc=ptc)
    pk_ii = pt.get_pt_pk2d(ccl_cosmo, ptt_i, ptc=ptc)

    # Set the parameters for our systematics
    systematics_params = ParamsMap(
        {
            "ia_a_1": a_1,
            "ia_a_2": a_2,
            "ia_a_d": a_d,
            "src0_delta_z": 0.000,
            "src1_delta_z": 0.003,
            "src2_delta_z": -0.001,
            "src3_delta_z": 0.002,
        }
    )

    # Apply the systematics parameters
    likelihood.update(systematics_params)
    # Compute the log-likelihood, using the ccl.Cosmology object as the input
    log_like = likelihood.compute_loglike(ccl_cosmo)

    print(f"Log-like = {log_like:.1f}")

    # Plot the predicted and measured statistic
    x = likelihood.statistics[0].ell_or_theta_
    y_data = likelihood.statistics[0].measured_statistic_

    assert likelihood.cov is not None

    y_err = np.sqrt(np.diag(likelihood.cov))[: len(x)]
    y_theory = likelihood.statistics[0].predicted_statistic_

    print(list(likelihood.statistics[0].cells.keys()))

    ells = likelihood.statistics[0].ells
    cells_gg = likelihood.statistics[0].cells[("shear", "shear")]
    cells_gi = likelihood.statistics[0].cells[("shear", "intrinsic_pt")]
    cells_ii = likelihood.statistics[0].cells[("intrinsic_pt", "intrinsic_pt")]
    cells_total = likelihood.statistics[0].cells["total"]

    # Code that computes effect from IA using that Pk2D object
    t_lens = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz))
    t_ia = ccl.WeakLensingTracer(
        ccl_cosmo,
        dndz=(z, nz),
        has_shear=False,
        ia_bias=(z, np.ones_like(z)),
        use_A_ia=False,
    )
    cl_GI = ccl.angular_cl(ccl_cosmo, t_lens, t_ia, ells, p_of_k_a=pk_im)
    cl_II = ccl.angular_cl(ccl_cosmo, t_ia, t_ia, ells, p_of_k_a=pk_ii)
    # The weak gravitational lensing power spectrum
    cl_GG = ccl.angular_cl(ccl_cosmo, t_lens, t_lens, ells)
    # The observed angular power spectrum is the sum of the two.
    cl_theory = (
        cl_GG + 2 * cl_GI + cl_II
    )  # normally we would also have a third term, +cl_II).

    # plt.plot(x, y_theory, label="Total")
    plt.plot(ells, cells_gg, label="GG firecrown")
    plt.plot(ells, cl_GG, ls="--", label="GG CCL")
    plt.plot(ells, -cells_gi, label="-GI firecrown")
    plt.plot(ells, -cl_GI, ls="--", label="-GI CCL")
    plt.plot(ells, cells_ii, label="II firecrown")
    plt.plot(ells, cl_II, ls="--", label="II CCL")
    plt.plot(ells, cells_total, label="total firecrown")
    plt.plot(ells, cl_theory, ls="--", label="total CCL")

    # plt.errorbar(x, y_data, y_err, ls="none", marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.legend()
    # plt.xlim(right=5e3)
    # plt.ylim(bottom=1e-12)
    plt.title("TATT IA")
    plt.savefig("plots/tatt.png", facecolor="white", dpi=300)

    plt.show()
