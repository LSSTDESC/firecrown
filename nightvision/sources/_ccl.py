from scipy.interpolate import Akima1DInterpolator
import pandas as pd
import pyccl as ccl

from .. import systematics as falcon_sys

__all__ = ['parse_ccl_source', 'build_ccl_source']


def parse_ccl_source(
        *,
        kind,
        data,
        has_intrinsic_alignment=False,
        systematics=None):
    """Parse a CCL source out of a config file.

    Parameters
    ----------
    kind : str
        One of 'ClTracerLensing' or 'ClTracerNumberCounts'.
    data : str
        The path to the data.
    has_intrinsic_alignment : bool, optional
        If the source has instrinsic alignments.
    systematics : dict of systematic descriptions or None
        A dict of systematic descriptions if any. See the functions in
        module `nightvision.systematics` for details and examples.

    Returns
    -------
    parsed : dict
        A dictionary with the input keys plus the keys
            'z': the redshift bin locations
            'n': the amplitudes of the redshift bins
            'pz_spline': a spline of the p(z) distribution (used for some
              photo-z systematics)
            'build_func': a function to call to transform the source by
              applying any systematics
    """
    new_keys = {
        'kind': getattr(ccl, kind),
        'data': data,
        'has_intrinsic_alignment': has_intrinsic_alignment,
        'systematics': systematics}

    df = pd.read_csv(data)
    _z, _nz = df['z'].values.copy(), df['nz'].values.copy()
    new_keys['z'] = _z
    new_keys['n'] = _nz
    new_keys['pz_spline'] = Akima1DInterpolator(_z, _nz)
    new_keys['build_func'] = build_ccl_source
    return new_keys


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
        build_func=None,
        data=None):
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
