import numpy as np
from ..wl import MultiplicativeShearBias


class DummySource(object):
    pass


def test_mult_shear_bias_smoke():
    src = DummySource()
    src.scale_ = 1.0
    m = 0.05
    params = {'blah': m}

    sys = MultiplicativeShearBias(m='blah')
    sys.apply(None, params, src)

    assert np.allclose(src.scale_, 1.05)
