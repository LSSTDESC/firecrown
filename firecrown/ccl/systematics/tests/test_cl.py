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
    # src = DummySource()
    # sys = PowerLawMOR()
    # sys.apply(None, None, src)
    #
    # assert hasattr(src, "mor_")
    # assert hasattr(src, "inv_mor_")
    # assert np.allclose(src.mor_(0.1, 0.5), 0.1)
    # assert np.allclose(src.inv_mor_(0.1, 0.5), 0.1)

    assert False
