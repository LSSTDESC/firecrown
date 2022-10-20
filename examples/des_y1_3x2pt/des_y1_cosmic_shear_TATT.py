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
n_source = 4
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
        sacc_tracer=f"src{i}",
        systematics=[ia_systematic]
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
pt_systematic = PTSystematic(with_NC=True, with_IA=True,
                             log10k_min=-4, log10k_max=2, nk_per_decade=20)
lk = ConstGaussian(statistics=list(stats.values()), systematics=[pt_systematic])

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

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()
    ccl_cosmo.compute_nonlin_power()
    # Set the parameters for our systematics
    systematics_params = ParamsMap({"ia_a_1": 1.0,
                                    "ia_a_2": 0.5,
                                    "ia_a_d": 0.5,

                                    "src0_delta_z": 0.001,
                                    "src1_delta_z": 0.003,
                                    "src2_delta_z": -0.001,
                                    "src3_delta_z": 0.002})

    # Apply the systematics parameters
    likelihood.update(systematics_params)
    # Compute the log-likelihood, using the ccl.Cosmology object as the input
    log_like = likelihood.compute_loglike(ccl_cosmo)

    print(f"Log-like = {log_like:.1f}")

    # Plot the predicted and measured statistic
    x = likelihood.statistics[0].ell_or_theta_
    y_data = likelihood.statistics[0].measured_statistic_
    y_err = np.sqrt(np.diag(likelihood.cov))[:len(x)]
    y_theory = likelihood.statistics[0].predicted_statistic_

    print(list(likelihood.statistics[0].cells.keys()))

    ells = likelihood.statistics[0].ells
    cells_gg = likelihood.statistics[0].cells[("delta_matter", "delta_matter")]
    cells_gi = likelihood.statistics[0].cells[("delta_matter", "intrinsic")]
    cells_ii = likelihood.statistics[0].cells[("intrinsic", "intrinsic")]

    # plt.plot(x, y_theory, label="Total")
    plt.plot(ells, cells_gg, label="GG")
    plt.plot(ells, cells_gi, label="GI")
    plt.plot(ells, cells_ii, label="II")
    # plt.errorbar(x, y_data, y_err, ls="none", marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.legend()
    plt.savefig("plots/tatt.png", facecolor="white", dpi=300)

    plt.show()
