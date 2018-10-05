import pyccl as ccl
from . import systematics as firecrown_sys


def build_ccl_source(
        *,
        cosmo,
        parameters,
        kind,
        z_n,
        n,
        pz_spline,
        has_intrinsic_alignment=False,
        systematics=None):
    """Build a CCL Tracer from a set of source keys.

    Parameters
    ----------
    cosmo : a `ccl.Cosmology` object
        The current cosmology.
    parameters : dict
        Dictionary mapping parameter names to values.
    kind : `ccl.ClTracer` or one of its subclasses
        The class to instantiate.
    z_n : array-like, shape (n_bins,)
        The photo-z bin locations.
    n : array-like, shape (n_bins,)
        The photo-z distribution.
    pz_spline : function
        A function that computers a interpoled values of the photo-z
        distribution at a given redshift.
    has_intrinsic_alignment : bool, optional
        If the source has instrinsic alignments.
    systematics : dict of systematic descriptions or None
        A dict of systematic descriptions if any. See the functions in
        module `nightvision.systematics` for details and examples.

    Returns
    -------
    tracer : a CCL ClTracer object
        The tracer.
    scale : float
        The scaling factor to apply to the source.
    """
    systematics = systematics or {}
    scale = 1.0

    for sys, sys_params in systematics.items():
        if sys == 'photoz_shift':
            n = firecrown_sys.photoz_shift(
                z_n,
                pz_spline,
                parameters[sys_params['delta_z']])
        elif sys == 'wl_mult_bias':
            scale = 1.0 + parameters[sys_params['m']]
        else:
            raise ValueError(
                "Systematic `%s` is not valid for tracer type `%s` for "
                "source!" % (sys, kind))

    if kind is ccl.ClTracerLensing:
        tracer = kind(
            cosmo,
            has_intrinsic_alignment=has_intrinsic_alignment,
            n=(z_n, n))
    else:
        raise ValueError(
            "CCL source kind '%s' for source is not "
            "recognized!" % (kind))

    return tracer, scale
