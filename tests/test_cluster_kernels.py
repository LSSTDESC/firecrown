"""Tests for the cluster kernel module."""

import numpy as np
import pytest

from firecrown.models.cluster.kernel import (
    Completeness,
    Purity,
    SpectroscopicRedshift,
    TrueMass,
)


def test_create_spectroscopic_redshift_kernel():
    srk = SpectroscopicRedshift()
    assert srk is not None


def test_create_mass_kernel():
    mk = TrueMass()
    assert mk is not None


def test_create_completeness_kernel():
    ck = Completeness()
    ck.ac_mc = 13.31
    ck.bc_mc = 0.2025
    ck.ac_nc = 0.38
    ck.bc_nc = 1.2634
    assert ck is not None
    assert ck.ac_mc == 13.31
    assert ck.bc_mc == 0.2025
    assert ck.ac_nc == 0.38
    assert ck.bc_nc == 1.2634


def test_create_purity_kernel():
    pk = Purity()
    pk.ap_nc = 3.9193
    pk.bp_nc = -0.3323
    pk.ap_rc = 1.1839
    pk.bp_rc = -0.4077
    assert pk is not None
    assert pk.ap_nc == 3.9193
    assert pk.bp_nc == -0.3323
    assert pk.ap_rc == 1.1839
    assert pk.bp_rc == -0.4077


def test_spec_z_distribution():
    srk = SpectroscopicRedshift()
    assert srk.distribution() == 1.0


def test_true_mass_distribution():
    tmk = TrueMass()
    assert tmk.distribution() == 1.0


@pytest.mark.precision_sensitive
def test_purity_distribution():
    pk = Purity()
    pk.ap_nc = 3.9193
    pk.bp_nc = -0.3323
    pk.ap_rc = 1.1839
    pk.bp_rc = -0.4077
    mass_proxy = np.linspace(0.0, 2.5, 10, dtype=np.float64)

    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
    mass_proxy_limits = (1.0, 10.0)

    truth = np.array(
        [
            0.00242882,
            0.03294582,
            0.3122527,
            0.85213252,
            0.98584893,
            0.99875485,
            0.99988632,
            0.99998911,
            0.99999891,
            0.99999988,
        ],
        dtype=np.float64,
    )

    purity = pk.distribution(z, mass_proxy, mass_proxy_limits).flatten()
    assert isinstance(purity, np.ndarray)
    for ref, true in zip(purity, truth):
        assert ref == pytest.approx(true, rel=1e-5, abs=0.0)


@pytest.mark.precision_sensitive
def test_purity_distribution_uses_mean():
    pk = Purity()
    pk.ap_nc = 3.9193
    pk.bp_nc = -0.3323
    pk.ap_rc = 1.1839
    pk.bp_rc = -0.4077
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
    mass_proxy = np.array([-1.0], dtype=np.float64)
    mass_proxy_limits = (0.0, 2.0)

    truth = np.array(
        [
            0.89705651,
            0.92238419,
            0.94154163,
            0.95593305,
            0.96670586,
            0.97476117,
            0.98078884,
            0.98530847,
            0.98870753,
            0.99127329,
        ],
        dtype=np.float64,
    )
    purity = pk.distribution(z, mass_proxy, mass_proxy_limits).flatten()
    assert isinstance(purity, np.ndarray)
    for ref, true in zip(purity, truth):
        assert ref == pytest.approx(true, rel=1e-7, abs=0.0)


@pytest.mark.precision_sensitive
def test_completeness_distribution():
    ck = Completeness()
    ck.ac_mc = 13.31
    ck.bc_mc = 0.2025
    ck.ac_nc = 0.38
    ck.bc_nc = 1.2634
    mass = np.linspace(13.0, 15.0, 10, dtype=np.float64)
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)

    truth = np.array(
        [
            0.10239024,
            0.19090539,
            0.35438466,
            0.58952617,
            0.80866296,
            0.93327968,
            0.98115635,
            0.99543348,
            0.99902667,
            0.99981606,
        ]
    )

    comp = ck.distribution(mass, z).flatten()
    assert isinstance(comp, np.ndarray)
    for ref, true in zip(comp, truth):
        assert ref == pytest.approx(true, rel=1e-7, abs=0.0)
