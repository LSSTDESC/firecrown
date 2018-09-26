import pyccl as ccl
from . import systematics as falcon_sys


def build_ccl_source(
        *,
        cosmo,
        params,
        src_name,
        kind,
        z,
        n,
        pz_spline,
        has_intrinsic_alignment=False,
        systematics=None,
        build_func=None):
    """Build a CCL Tracer from a set of source keys.

    Parameters
    ----------
    TODO: Write shit here.

    Returns
    -------
    tracer : a CCL Tracer
    """
    systematics = systematics or {}

    for sys, sys_params in systematics.items():
        if sys == 'photoz_shift':
            n = falcon_sys.photoz_shift(
                z,
                pz_spline,
                params[sys_params['delta_z']])
        else:
            raise ValueError(
                "Systematic `%s` is not valid for tracer type `%s` for "
                "source '%s'!" % (sys, kind, src_name))

    if kind is ccl.ClTracerLensing:
        tracer = kind(
            cosmo,
            has_intrinsic_alignment=has_intrinsic_alignment,
            n=(z, n))
    else:
        raise ValueError(
            "CCL source kind '%s' for source '%s' not "
            "recognized!" % (kind, src_name))

    return tracer
