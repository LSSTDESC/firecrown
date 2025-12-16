import numpy as np
import pytest

from firecrown.metadata_types import CMBLensing, TomographicBin
from firecrown.metadata_types import CMB, Galaxies


def make_tomobin() -> TomographicBin:
    z = np.array([0.0, 0.5, 1.0])
    dndz = np.array([0.1, 0.2, 0.7])
    return TomographicBin(bin_name="src0", z=z, dndz=dndz, measurements={Galaxies.COUNTS})


def test_cmblensing_basic():
    cmb = CMBLensing(bin_name="cmb0", z_lss=1090.0)
    assert cmb.bin_name == "cmb0"
    assert cmb.z_lss == 1090.0
    assert cmb.measurements == {CMB.CONVERGENCE}


def test_cmblensing_validation():
    with pytest.raises(ValueError):
        CMBLensing(bin_name="", z_lss=1090.0)

    with pytest.raises(ValueError):
        CMBLensing(bin_name="cmb0", z_lss=float("nan"))

    with pytest.raises(ValueError):
        CMBLensing(bin_name="cmb0", z_lss=-1.0)


def test_tomographicbin_equality_and_cmb_mismatch():
    tb1 = make_tomobin()
    tb2 = TomographicBin(
        bin_name="src0",
        z=np.array([0.0, 0.5, 1.0]),
        dndz=np.array([0.1, 0.2, 0.7]),
        measurements={Galaxies.COUNTS},
    )

    assert tb1 == tb2

    cmb = CMBLensing(bin_name="cmb0", z_lss=1090.0)
    # Different tracer implementations should not be equal
    assert (tb1 == cmb) is False
    assert (cmb == tb1) is False

    # Unrelated types should not be considered equal
    assert (tb1 == object()) is False
    assert (cmb == object()) is False


def test_cmblensing_equality():
    c1 = CMBLensing(bin_name="cmb0", z_lss=1090.0)
    c2 = CMBLensing(bin_name="cmb0", z_lss=1090.0)
    assert c1 == c2

    c3 = CMBLensing(bin_name="cmb1", z_lss=1090.0)
    assert c1 != c3
