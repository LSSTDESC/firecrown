from ..cl import MORMurata


class DummySource(object):
    def integrate_pmor_dz_dm_dproxy(self, *args, **kwargs):
        return 1.05


def test_mor_murata_smoke():
    src = DummySource()
    src.bias_ = 1.0

    syst = MORMurata(
        mor_a='a', mor_b='b', mor_c='d',
        mor_scatter_s0='y', mor_scatter_qm='r',
        mor_scatter_qz='t')

    syst.apply(None, None, src)
    assert src.bias_ == 1.05
