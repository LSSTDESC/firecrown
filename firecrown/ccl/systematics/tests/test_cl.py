import numpy as np

from ..cl import (
    IdentityFunctionMOR,
    TopHatSelectionFunction,
    PowerLawMOR,
)


class DummySource(object):
    pass


def test_identity_function_mor_systematic():
    src = DummySource()
    sys = IdentityFunctionMOR()
    sys.apply(None, None, src)

    assert hasattr(src, "mor_")
    assert hasattr(src, "inv_mor_")
    assert np.allclose(src.mor_(0.1, 0.5), 0.1)
    assert np.allclose(src.inv_mor_(0.1, 0.5), 0.1)


def test_powerlaw_mor_systematic():
    src = DummySource()
    sys = PowerLawMOR(lnlam_norm="nrm", mass_slope="ms", a_slope="as")
    params = {
        "nrm": 2.1,
        "ms": 0.4,
        "as": 1.5,
    }
    sys.apply(None, params, src)

    assert hasattr(src, "mor_")
    assert hasattr(src, "inv_mor_")
    assert np.allclose(
        src.mor_(0.1, 0.5),
        2.1 + 0.4 * (0.1 - np.log(1e14)) + 1.5 * np.log(0.5),
    )
    assert np.allclose(
        src.inv_mor_(-10.5, 0.5),
        (-10.5 - 2.1 - 1.5 * np.log(0.5)) / 0.4 + np.log(1e14),
    )


def test_tophat_selection_function_systematic():
    def dndz_interp_(z):
        return 0.5/z

    src = DummySource()
    src.lnlam_min_ = -12
    src.lnlam_max_ = -10
    src.dndz_interp_ = dndz_interp_
    sys = PowerLawMOR(lnlam_norm="nrm", mass_slope="ms", a_slope="as")
    params = {
        "nrm": 2.1,
        "ms": 0.4,
        "as": 1.5,
    }
    sys.apply(None, params, src)

    sel = TopHatSelectionFunction()
    sel.apply(None, None, src)

    assert hasattr(src, "selfunc_")

    lnm_min = src.inv_mor_(src.lnlam_min_, 0.5)
    lnm_max = src.inv_mor_(src.lnlam_max_, 0.5)

    assert np.array_equal(
        src.selfunc_((lnm_min + lnm_max)/2, 0.5),
        np.atleast_2d(0.5 / (1/0.5 - 1)),
    )

    assert np.array_equal(src.selfunc_(lnm_min - 10, 0.5), np.atleast_2d(0))
    assert np.array_equal(src.selfunc_(lnm_max + 10, 0.5), np.atleast_2d(0))
