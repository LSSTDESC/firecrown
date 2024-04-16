"""The DES Y1 3x2pt likelihood factory module."""

import os
import sacc

import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
import firecrown.likelihood.gauss_family.statistic.source.number_counts as nc
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian


# The likelihood used for DES Y1 3x2pt analysis is a Gaussian likelihood, which
# necessitates providing a list of statistics that are each represented by a two-point
# function. To construct the two-point function, a list of sources is required, with
# each source being responsible for computing the theoretical prediction for a specific
#  segment of the data. These sources are created using the build_likelihood function
# and also contain a list of systematics. The systematics are classes that modify the
# theoretical prediction and are also constructed in the build_likelihood function.
def build_likelihood(_):
    """Build the DES Y1 3x2pt likelihood."""
    # Creates a LAI systematic. This is a systematic that is applied to
    # all weak-lensing probes. The `sacc_tracer` argument is used to identify the
    # section of the SACC file that this systematic will be applied to. In this
    # case we want to apply it to all weak-lensing probes, so we use the
    # empty string.
    lai_systematic = wl.LinearAlignmentSystematic(sacc_tracer="")

    # Creating sources, each one maps to a specific section of a SACC file. In
    # this case src0, src1, src2 and src3 describe weak-lensing probes. The
    # sources are saved in a dictionary since they will be used by one or more
    # two-point function.
    sources: dict[str, wl.WeakLensing | nc.NumberCounts] = {}

    for i in range(4):
        # Each weak-lensing section has its own multiplicative bias. Parameters
        # reflect this by using src{i}_ prefix.
        mbias = wl.MultiplicativeShearBias(sacc_tracer=f"src{i}")

        # We also include a photo-z shift bias (a constant shift in dndz). We
        # also have a different parameter for each bin, so here again we use the
        # src{i}_ prefix.
        wl_pzshift = wl.PhotoZShift(sacc_tracer=f"src{i}")

        # Now we can finally create the weak-lensing source that will compute the
        # theoretical prediction for that section of the data, given the
        # systematics.
        sources[f"src{i}"] = wl.WeakLensing(
            sacc_tracer=f"src{i}", systematics=[lai_systematic, mbias, wl_pzshift]
        )

    # Creating the number counting sources. There are five sources each one
    # labeled by lens{i}.
    for i in range(5):
        # We also include a photo-z shift for the dndz.
        nc_pzshift = nc.PhotoZShift(sacc_tracer=f"lens{i}")

        # The source is created and saved (temporarily in the sources dict).
        sources[f"lens{i}"] = nc.NumberCounts(
            sacc_tracer=f"lens{i}", systematics=[nc_pzshift], derived_scale=True
        )

    # Now that we have all sources we can instantiate all the two-point
    # functions. The weak-lensing sources have two "data types", for each one we
    # create a new two-point function.
    stats = {}
    for stat, sacc_stat in [
        ("xip", "galaxy_shear_xi_plus"),
        ("xim", "galaxy_shear_xi_minus"),
    ]:
        # Creating all auto/cross-correlations two-point function objects for the
        # weak-lensing probes.
        for i in range(4):
            for j in range(i, 4):
                stats[f"{stat}_src{i}_src{j}"] = TwoPoint(
                    source0=sources[f"src{i}"],
                    source1=sources[f"src{j}"],
                    sacc_data_type=sacc_stat,
                )
    # The following two-point function objects compute the cross correlations
    # between the weak-lensing and number count sources.
    for j in range(5):
        for i in range(4):
            stats[f"gammat_lens{j}_src{i}"] = TwoPoint(
                source0=sources[f"lens{j}"],
                source1=sources[f"src{i}"],
                sacc_data_type="galaxy_shearDensity_xi_t",
            )

    # Finally the instances for the lensing auto-correlations are created.
    for i in range(5):
        stats[f"wtheta_lens{i}_lens{i}"] = TwoPoint(
            source0=sources[f"lens{i}"],
            source1=sources[f"lens{i}"],
            sacc_data_type="galaxy_density_xi",
        )

    # Here we instantiate the actual likelihood. The statistics argument carry
    # the order of the data/theory vector.
    likelihood = ConstGaussian(statistics=list(stats.values()))

    # We load the correct SACC file.
    saccfile = os.path.expanduser(
        os.path.expandvars(
            "${FIRECROWN_DIR}/examples/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits"
        )
    )
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # The read likelihood method is called passing the loaded SACC file, the
    # two-point functions will receive the appropriated sections of the SACC
    # file and the sources their respective dndz.
    likelihood.read(sacc_data)

    # This script will be loaded by the appropriated connector. The framework
    # will call the factory function that should return a Likelihood instance.
    return likelihood
