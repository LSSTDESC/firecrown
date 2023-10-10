from firecrown.models.kernel import Completeness, Purity, KernelType, Kernel
from firecrown.models.mass_observable import Mass, MassRichnessMuSigma
from firecrown.models.redshift import (
    Redshift,
    SpectroscopicRedshift,
    DESY1PhotometricRedshift,
)


def test_desy1_photometric_redshift_kernel():
    drk = DESY1PhotometricRedshift()
    assert isinstance(drk, Kernel)
    assert drk.kernel_type == KernelType.z_proxy
    assert drk.is_dirac_delta is True
    assert drk.integral_bounds is None

    drk = DESY1PhotometricRedshift([(0, 1)])
    assert drk.is_dirac_delta is False
    assert len(drk.integral_bounds) == 1
    assert drk.integral_bounds[0] == (0, 1)


def test_spectroscopic_redshift_kernel():
    srk = SpectroscopicRedshift()
    assert isinstance(srk, Kernel)
    assert srk.kernel_type == KernelType.z_proxy
    assert srk.is_dirac_delta is True
    assert srk.integral_bounds is None

    srk = SpectroscopicRedshift([(0, 1)])
    assert srk.is_dirac_delta is False
    assert len(srk.integral_bounds) == 1
    assert srk.integral_bounds[0] == (0, 1)


def test_redshift_kernel():
    rk = Redshift()
    assert isinstance(rk, Kernel)
    assert rk.kernel_type == KernelType.z
    assert rk.is_dirac_delta is True
    assert rk.integral_bounds is None

    rk = Redshift([(0, 1)])
    assert rk.is_dirac_delta is False
    assert len(rk.integral_bounds) == 1
    assert rk.integral_bounds[0] == (0, 1)


def test_musigma_kernel():
    msk = MassRichnessMuSigma(1, 1)
    assert isinstance(msk, Kernel)
    assert msk.kernel_type == KernelType.mass_proxy
    assert msk.is_dirac_delta is True
    assert msk.integral_bounds is None

    msk = MassRichnessMuSigma(1, 1, integral_bounds=[(0, 1)])
    assert msk.is_dirac_delta is False
    assert len(msk.integral_bounds) == 1
    assert msk.integral_bounds[0] == (0, 1)


def test_mass_kernel():
    mk = Mass()
    assert isinstance(mk, Kernel)
    assert mk.kernel_type == KernelType.mass
    assert mk.is_dirac_delta is True
    assert mk.integral_bounds is None

    mk = Mass([(0, 1)])
    assert mk.is_dirac_delta is False
    assert len(mk.integral_bounds) == 1
    assert mk.integral_bounds[0] == (0, 1)


def test_completeness_kernel():
    ck = Completeness()
    assert isinstance(ck, Kernel)
    assert ck.kernel_type == KernelType.completeness
    assert ck.is_dirac_delta is True
    assert ck.integral_bounds is None

    ck = Completeness([(0, 1)])
    assert ck.is_dirac_delta is False
    assert len(ck.integral_bounds) == 1
    assert ck.integral_bounds[0] == (0, 1)


def test_purity_kernel():
    pk = Purity()
    assert isinstance(pk, Kernel)
    assert pk.kernel_type == KernelType.purity
    assert pk.is_dirac_delta is True
    assert pk.integral_bounds is None

    pk = Purity([(0, 1)])
    assert pk.is_dirac_delta is False
    assert len(pk.integral_bounds) == 1
    assert pk.integral_bounds[0] == (0, 1)
